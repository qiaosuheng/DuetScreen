"""
Real-time processing dataset - Fully using Nucleotide Transformer v3
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from typing import List, Dict, Optional, Any
import random
from concurrent.futures import ThreadPoolExecutor
import threading
from .reproducibility import seed_data_worker


def variant_indices(seq_list, batch_size):
    """Indices of records whose variant sequence is non-empty.

    With symmetric augmentation, negatives also carry semantic/confusion
    views (label 0); losses and metrics follow these indices instead of
    positive-only selection.
    """
    if not seq_list:
        return []
    return [i for i in range(min(batch_size, len(seq_list))) if seq_list[i]]


class RealtimeSequenceDataset(Dataset):
    """Real-time sequence dataset - Fully using Nucleotide Transformer v3"""
    
    def __init__(
        self,
        segments: List[Dict],
        max_seq_len: int = 512,
        use_cache: bool = True,
        pure_task: bool = False
    ):
        """
        Initialize real-time dataset
        
        Args:
            segments: List of segment data
            max_seq_len: Maximum sequence length (number of DNA bases)
            use_cache: Whether to use cache (set to False for real-time processing)
        """
        self.segments = segments
        self.max_seq_len = max_seq_len
        self.use_cache = use_cache
        # Arm C (pure-task protocol): never expose variant views, even when the
        # underlying records carry them.
        self.pure_task = pure_task
        
        # Cache system (optional)
        self.sequence_cache = {} if use_cache else None
        self.cache_lock = threading.Lock()
        
        print(f"📊 Nucleotide Transformer v3 Dataset Initialization")
        print(f"  Total segments: {len(segments):,}")
        print(f"  Real-time processing: {not use_cache}")
        print(f"  Maximum sequence length (DNA bases): {max_seq_len}")
    
    def __len__(self) -> int:
        return len(self.segments)
        
    # In the _get_sequence method of RealtimeSequenceDataset class, update sequence processing:

    def _get_sequence(self, seq: str, cache_key: str = None) -> str:
        """Get sequence (with cache support) - only clean characters, no padding"""
        if self.use_cache and cache_key:
            with self.cache_lock:
                if cache_key in self.sequence_cache:
                    return self.sequence_cache[cache_key]
        
        # Ensure sequence only contains valid characters
        valid_chars = set('ACGTNacgtn')
        seq = ''.join([c for c in seq if c in valid_chars])
        
        # Convert to uppercase
        seq = seq.upper()
        
        # If sequence is too short, add some N's to give it basic length
        if len(seq) < 50:
            seq = seq + 'N' * (50 - len(seq))
        
        # Cache result
        if self.use_cache and cache_key:
            with self.cache_lock:
                self.sequence_cache[cache_key] = seq
        
        return seq

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seg = self.segments[idx]
        
        # Get original sequence
        original_seq = seg.get('original_seq', '')
        if not original_seq:
            raise ValueError(f"Sample {idx} missing original sequence")
        
        # Get label
        label = seg.get('original_label', seg.get('label', 0.0))
        
        # Process sequence length
        original_seq = self._get_sequence(original_seq, f"orig_{idx}")
        
        # Build return item
        item = {
            'sequence': original_seq,  # Original DNA sequence
            'labels': torch.FloatTensor([float(label)]),
            'segment_id': seg.get('segment_id', f"seq_{idx}"),
        }
        
        if self.pure_task:
            # Pure-task protocol: originals only.
            return item

        # Process variant sequences (if they exist)
        # Semantic variants
        if 'positive_views' in seg and seg['positive_views']:
            semantic_view = None
            confusion_view = None
            
            # Find semantic and confusion variants
            for i, view_type in enumerate(seg.get('view_types', [])):
                if i < len(seg['positive_views']):
                    if view_type == 'semantic':
                        semantic_view = seg['positive_views'][i]
                    elif view_type == 'confusion':
                        confusion_view = seg['positive_views'][i]
            
            if semantic_view:
                semantic_seq = self._get_sequence(semantic_view, f"semantic_{idx}")
                item['semantic_sequence'] = semantic_seq
            
            if confusion_view:
                confusion_seq = self._get_sequence(confusion_view, f"confusion_{idx}")
                item['confusion_sequence'] = confusion_seq
        
        # Random mutation variants
        if 'contrastive_negative_views' in seg and seg['contrastive_negative_views']:
            random_view = random.choice(seg['contrastive_negative_views'])
            random_seq = self._get_sequence(random_view, f"random_{idx}_{hash(random_view)}")
            item['random_mutation_sequence'] = random_seq
        
        return item
    
    def clear_cache(self):
        """Clear cache"""
        if self.sequence_cache:
            self.sequence_cache.clear()


def collate_nucleotide_features(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for Nucleotide Transformer v3 features"""
    if not batch:
        return {}
    
    batch_size = len(batch)
    
    # Collect all sequences
    sequences = []
    labels = []
    segment_ids = []
    
    semantic_sequences = []
    confusion_sequences = []
    random_mutation_sequences = []
    
    for item in batch:
        sequences.append(item['sequence'])
        labels.append(item['labels'])
        segment_ids.append(item.get('segment_id', 'unknown'))
        
        # Variant sequences
        semantic_sequences.append(item.get('semantic_sequence', None))
        confusion_sequences.append(item.get('confusion_sequence', None))
        random_mutation_sequences.append(item.get('random_mutation_sequence', None))
    
    # Stack labels
    labels_tensor = torch.stack(labels).squeeze()
    
    # Build result
    result = {
        'sequences': sequences,  # List[str] - original sequences
        'labels': labels_tensor,
        'segment_ids': segment_ids,
    }
    
    # Process variant sequences - ensure consistent list length
    def process_variant_sequences(variant_list):
        # Find indices with variants
        valid_indices = [i for i, seq in enumerate(variant_list) if seq is not None]
        
        if valid_indices:
            # Create complete list, replace None with empty string
            processed = []
            for seq in variant_list:
                if seq is not None:
                    processed.append(seq)
                else:
                    processed.append("")  # Empty string indicates no variant
            return processed
        return None
    
    result['semantic_sequences'] = process_variant_sequences(semantic_sequences)
    result['confusion_sequences'] = process_variant_sequences(confusion_sequences)
    result['random_mutation_sequences'] = process_variant_sequences(random_mutation_sequences)
    
    return result


def create_nucleotide_dataloader(
    segments: List[Dict],
    batch_size: int = 8,  # Reduce batch size to accommodate NTv3
    max_seq_len: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    use_cache: bool = True,
    seed: Optional[int] = None,
    pure_task: bool = False,
) -> DataLoader:
    """Create a loader with an optional independent shuffle/worker RNG.

    ``seed`` controls loading order and worker randomness, not dataset membership.
    Passing None preserves the legacy use of PyTorch's global generator.
    """

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    
    # Create dataset
    dataset = RealtimeSequenceDataset(
        segments=segments,
        max_seq_len=max_seq_len,
        use_cache=use_cache,
        pure_task=pure_task
    )
    
    print(f"🔧 Creating Nucleotide Transformer v3 data loader:")
    print(f"  Batch size: {batch_size} ⚠️ (NTv3 requires large memory)")
    print(f"  Dataset size: {len(dataset):,}")
    print(f"  Shuffle: {shuffle}")
    print(f"  Number of workers: {num_workers}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=seed_data_worker if seed is not None else None,
        pin_memory=True,
        collate_fn=collate_nucleotide_features,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
