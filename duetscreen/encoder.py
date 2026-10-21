"""
Fully Trainable Nucleotide Transformer v3 (NTv3_8M_pre) Embedding Layer
Supports freezing NTv3 parameters, internal configuration only unfreezes deconv_tower_blocks module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Optional, Dict, Any, Tuple
import os
import hashlib
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class NTv3TrainableEmbedding(nn.Module):
    """NTv3_8M_pre trainable embedding layer, internal configuration only unfreezes deconv_tower_blocks"""
    
    def __init__(
        self,
        model_repo: str = "InstaDeepAI/NTv3_8M_pre",
        output_dim: int = 256,  # Dimension to output to Mamba
        max_seq_len: int = 2048,  # Number of DNA bases
        use_cache: bool = False,  # Force disable caching to ensure dimension consistency
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        freeze_transformer: bool = False  # Whether to freeze NTv3 parameters
    ):
        super().__init__()
        
        self.model_repo = model_repo
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.use_cache = use_cache
        self.trust_remote_code = trust_remote_code
        self.freeze_transformer = freeze_transformer  # Store freeze state
        
        # Internal configuration: only unfreeze deconv_tower_blocks module
        # You can modify which modules to unfreeze here
        self._unfreeze_target_modules = ["deconv_tower_blocks"]
        
        # Auto-select device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🔍 Loading Nucleotide Transformer v3: {model_repo}")
        print(f"  Device: {self.device}")
        print(f"  Trust remote code: {trust_remote_code}")
        print(f"  Maximum sequence length: {max_seq_len}")
        print(f"  Freeze NTv3: {freeze_transformer}")
        
        try:
            # 1. Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_repo,
                trust_remote_code=trust_remote_code
            )
            
            # Print tokenizer information
            print(f"  Tokenizer vocabulary size: {len(self.tokenizer)}")
            print(f"  Padding token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
            
            # 2. Load full model
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_repo,
                trust_remote_code=trust_remote_code
            ).to(self.device)
            
            print(f"✅ Model loaded successfully")
            
            # NTv3_8M_pre hidden layer dimension is 256
            self.hidden_size = 256
            print(f"  NTv3 hidden size: {self.hidden_size} (fixed)")

            # Print basic structure
            self.print_basic_info()

            # Set whether model parameters are trainable based on freeze_transformer parameter
            if self.freeze_transformer:
                # Freeze parameters
                for param in self.model.parameters():
                    param.requires_grad = False
                
                # Then unfreeze specified modules
                if self._unfreeze_target_modules:
                    self._unfreeze_specific_modules()
                    print(f"  🔥 Unfrozen specified modules: {self._unfreeze_target_modules}")
                else:
                    print(f"  ❄️  NTv3 model parameters frozen")
            else:
                print(f"  🔥 NTv3 model parameters trainable (fully fine-tuned)")
            
            # Model architecture information
            print(f"📋 Model architecture: U-Net style conv tower → Transformer stack → deconv tower → LM head")
            print(f"  Input requirement: Sequence length must be multiple of 128")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
        
        # Cache system (disabled to ensure feature dimension consistency)
        self.cache = None  # Force disable cache
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Adaptive pooling layer: unify model output to fixed length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(max_seq_len)
        
        # Projection layer: project model output to our required dimension
        if self.hidden_size != output_dim:
            self.feature_projection = nn.Sequential(
                nn.Linear(self.hidden_size, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ).to(self.device)
            print(f"  Projection layer enabled: {self.hidden_size} -> {output_dim}")
        else:
            self.feature_projection = nn.Identity()
            print(f"  No projection layer (hidden layer={output_dim})")
        
        print(f"📊 NTv3 embedding configuration:")
        print(f"  Model repository: {model_repo}")
        print(f"  Original hidden size: {self.hidden_size}")
        print(f"  Output dimension: {output_dim}")
        print(f"  Maximum sequence length (bases): {max_seq_len}")
        print(f"  Use cache: {use_cache}")
        print(f"  Freeze parameters: {freeze_transformer}")
        if self.freeze_transformer and self._unfreeze_target_modules:
            print(f"  Unfrozen modules: {self._unfreeze_target_modules}")
        
        # Print parameter statistics
        self.print_param_stats()
    
    def _unfreeze_specific_modules(self):
        """Unfreeze specific modules (internal method)"""
        if not self._unfreeze_target_modules:
            return
        
        total_unfrozen = 0
        
        # Iterate through all modules
        for module_name in self._unfreeze_target_modules:
            module_found = False
            
            # Recursively search modules
            for name, module in self.model.named_modules():
                if module_name in name:
                    # Found module, unfreeze all its parameters
                    for param in module.parameters():
                        param.requires_grad = True
                        total_unfrozen += param.numel()
                    module_found = True
                    print(f"    Found and unfrozen module: {name}")
            
            if not module_found:
                print(f"  ⚠️  Warning: Module '{module_name}' not found")
        
        print(f"  ✅ Total unfrozen parameters: {total_unfrozen:,}")
    
    def print_param_stats(self):
        """Print parameter statistics"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Count parameters by module
        print("\n📈 NTv3 module parameter statistics:")
        print("-" * 40)
        
        # Count parameters for main modules
        module_stats = {}
        for name, module in self.model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                module_stats[name] = module_params
        
        # Print module parameters
        for module_name, params in module_stats.items():
            trainable = sum(p.numel() for p in getattr(self.model, module_name).parameters() if p.requires_grad)
            status = "Trainable" if trainable > 0 else "Frozen"
            print(f"  {module_name:20s}: {params:12,} parameters ({status})")
        
        print("-" * 40)
        
        # Projection layer parameters
        proj_params = sum(p.numel() for p in self.feature_projection.parameters())
        
        print(f"\nTotal:")
        print(f"  NTv3 total parameters: {total_params:,}")
        print(f"  NTv3 trainable: {trainable_params:,}")
        print(f"  Projection layer parameters: {proj_params:,}")
        
        if total_params > 0:
            trainable_ratio = trainable_params / total_params * 100
            print(f"  NTv3 trainable ratio: {trainable_ratio:.1f}%")
    
    def _preprocess_sequence(self, sequence: str) -> str:
        """Preprocess DNA sequence - ensure sequence only contains valid characters"""
        # Ensure sequence only contains valid characters
        valid_chars = set('ACGTNacgtn')
        sequence = ''.join([c for c in sequence if c in valid_chars])
        
        if not sequence:
            return 'N' * 100  # Return minimum length sequence
        
        # Convert to uppercase
        return sequence.upper()
    
    def _tokenize_sequences(self, sequences: List[str]) -> torch.Tensor:
        """Tokenize sequences - return only input_ids"""
        # Preprocess all sequences (only clean characters)
        processed_seqs = [self._preprocess_sequence(seq) for seq in sequences]
        
        # Use tokenizer for tokenization and padding
        batch = self.tokenizer(
            processed_seqs,
            add_special_tokens=False,  # Don't add special tokens
            padding=True,  # Enable padding
            pad_to_multiple_of=128,  # Pad to multiple of 128
            max_length=self.max_seq_len,  # Maximum length
            truncation=True,  # Truncate if exceeding length
            return_tensors="pt"
        )
        
        # Return only input_ids
        input_ids = batch['input_ids']
        
        return input_ids
    
    def _process_model_output(self, features: torch.Tensor) -> torch.Tensor:
        """Process model output, use adaptive pooling to unify feature length"""
        batch_size, seq_len, hidden_dim = features.shape
        
        # NTv3 output feature length may not be max_seq_len, need to unify
        if seq_len != self.max_seq_len:
            # Use adaptive average pooling to unify to max_seq_len length
            # First transpose dimensions: [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim, seq_len]
            features_t = features.transpose(1, 2)
            
            # Adaptive pooling to max_seq_len length
            features_pooled = self.adaptive_pool(features_t)
            
            # Transpose back: [batch_size, hidden_dim, max_seq_len] -> [batch_size, max_seq_len, hidden_dim]
            features = features_pooled.transpose(1, 2)
            
            # Verify length is now correct
            new_seq_len = features.shape[1]
            if new_seq_len != self.max_seq_len:
                print(f"⚠️  Length still mismatched after adaptive pooling: {new_seq_len} != {self.max_seq_len}")
        
        # Project to target dimension
        features = self.feature_projection(features)
        
        return features
    
    def extract_features(self, sequences: List[str], training_mode: bool = True) -> torch.Tensor:
        """
        Extract features - no cache usage, ensure dimension consistency
        
        Args:
            sequences: List of DNA sequences
            training_mode: Whether in training mode
        """
        if not sequences:
            return torch.empty(0, self.max_seq_len, self.output_dim, device=self.device)
        
        # Tokenize all sequences
        input_ids = self._tokenize_sequences(sequences)
        input_ids = input_ids.to(self.device)
        
        # Decide whether to compute gradients based on freeze state and training mode
        # Check if any parameters require gradients
        has_trainable_params = any(p.requires_grad for p in self.model.parameters())
        
        if training_mode and has_trainable_params:
            # Training mode and NTv3 has trainable parameters: compute gradients
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        else:
            # Evaluation mode or NTv3 frozen: do not compute gradients
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        
        # Extract features: use last layer hidden states
        hidden_states = outputs.hidden_states
        features = hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Process features, use adaptive pooling to ensure dimension consistency
        features = self._process_model_output(features)
        
        return features
    
    def forward(self, sequences: List[str], training_mode: bool = True) -> torch.Tensor:
        """Forward pass"""
        return self.extract_features(sequences, training_mode=training_mode)
    
    def clear_cache(self):
        """Clear cache"""
        print("ℹ️  Cache disabled")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_enabled': False,
            'cache_size': 0,
            'hit_rate': 0.0
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        vocab_size = len(self.tokenizer) if hasattr(self, 'tokenizer') else 'N/A'
        
        return {
            'model_repo': self.model_repo,
            'vocab_size': vocab_size,
            'hidden_size': self.hidden_size,
            'output_dim': self.output_dim,
            'max_seq_len': self.max_seq_len,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'freeze_transformer': self.freeze_transformer,
            'unfreeze_modules': self._unfreeze_target_modules if self.freeze_transformer else [],
            'cache_enabled': False,
            'pad_token': self.tokenizer.pad_token,
            'pad_token_id': self.tokenizer.pad_token_id,
            'adaptive_pool': True,
            'pool_target_length': self.max_seq_len
        }
    
    def print_model_summary(self):
        """Print model summary"""
        info = self.get_model_info()
        print(f"🧬 NTv3 Embedding Model Summary")
        print(f"  =========================================")
        print(f"  Model repository: {info['model_repo']}")
        print(f"  Vocabulary size: {info['vocab_size']}")
        print(f"  Hidden size: {info['hidden_size']} (NTv3_8M_pre)")
        print(f"  Output dimension: {info['output_dim']}")
        print(f"  Maximum sequence length: {info['max_seq_len']}")
        print(f"  Freeze NTv3: {info['freeze_transformer']}")
        if info['freeze_transformer'] and info['unfreeze_modules']:
            print(f"  Unfrozen modules: {info['unfreeze_modules']}")
        print(f"  Total parameters: {info['total_params']:,}")
        print(f"  Trainable parameters: {info['trainable_params']:,}")
        if info['total_params'] > 0:
            trainable_ratio = info['trainable_params'] / info['total_params'] * 100
            print(f"  Trainable ratio: {trainable_ratio:.1f}%")
        print(f"  =========================================")
    
    def set_freeze_state(self, freeze: bool):
        """Dynamically set freeze state"""
        if freeze != self.freeze_transformer:
            self.freeze_transformer = freeze
            
            if freeze:
                # Freeze all parameters
                for param in self.model.parameters():
                    param.requires_grad = False
                
                # Then unfreeze specified modules
                if self._unfreeze_target_modules:
                    self._unfreeze_specific_modules()
            else:
                # Unfreeze all parameters
                for param in self.model.parameters():
                    param.requires_grad = True
            
            state = "Frozen" if freeze else "Unfrozen"
            print(f"🔄 NTv3 parameters {state}")
            if freeze and self._unfreeze_target_modules:
                print(f"  Unfrozen modules: {self._unfreeze_target_modules}")
            
            self.print_param_stats()
    
    def set_unfreeze_modules(self, modules: List[str]):
        """Dynamically set modules to unfreeze (internal use)"""
        self._unfreeze_target_modules = modules
        if self.freeze_transformer:
            print(f"🔄 Updated unfrozen modules: {modules}")
            # Reapply freeze/unfreeze state
            self.set_freeze_state(self.freeze_transformer)

    def print_basic_info(self):
        """Print simplified NTv3 network layer structure"""
        print("🧬 NTv3 Network Layer Structure (Simplified)")
        print("=" * 50)
        
        print(f"Model type: {type(self.model).__name__}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Output dimension: {self.output_dim}")
        
        print("\nMain modules:")
        print("-" * 30)
        
        # Only print main modules
        for name, module in self.model.named_children():
            num_params = sum(p.numel() for p in module.parameters())
            num_params_str = f"{num_params:,}" if num_params > 0 else "0"
            
            # Check if there are child modules
            children = list(module.children())
            if children:
                print(f"├─ {name} ({type(module).__name__}): {num_params_str} parameters")
                # Print first level child modules
                for child_name, child_module in module.named_children():
                    child_params = sum(p.numel() for p in child_module.parameters())
                    if child_params > 0:
                        print(f"│  └─ {child_name}: {child_params:,} parameters")
            else:
                print(f"└─ {name}: {num_params_str} parameters")
        
        print("\nParameter statistics:")
        print("-" * 30)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable: {trainable_params:,}")
        print(f"Frozen: {total_params - trainable_params:,}")
        
        print("=" * 50)
        
        # Display internal configuration
        if self.freeze_transformer and self._unfreeze_target_modules:
            print(f"Internal configuration: NTv3 frozen, only unfreeze {self._unfreeze_target_modules} modules")
        elif self.freeze_transformer:
            print(f"Internal configuration: NTv3 all parameters frozen")
        else:
            print(f"Internal configuration: NTv3 fully unfrozen")


# Factory function
def create_ntv3_trainable_embedding(config: dict) -> NTv3TrainableEmbedding:
    """Create NTv3 fully trainable embedding layer"""
    return NTv3TrainableEmbedding(
        model_repo=config.get('transformer_model_repo', 'InstaDeepAI/NTv3_8M_pre'),
        output_dim=config.get('embedding_dim', 256),
        max_seq_len=config.get('max_seq_len', 2048),
        use_cache=config.get('use_caching', False),  # Force disable cache
        device=config.get('device', None),
        trust_remote_code=config.get('trust_remote_code', True),
        freeze_transformer=config.get('freeze_transformer', False)
    )
