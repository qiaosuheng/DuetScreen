"""
Model prediction script
Supports custom classification threshold
"""

import torch
import numpy as np
_np_trapz = getattr(np, "trapz", None) or np.trapezoid
import os
import json
import argparse
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
from Bio import SeqIO
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)

# Import project modules
try:
    from .config import config
    from .model import create_nucleotide_mamba_model
    from .encoder import NTv3TrainableEmbedding
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please ensure the following files are in Python path:")
    print("1. config.py")
    print("2. model.py")
    print("3. nucleotide_v3_trainable.py")
    IMPORT_SUCCESS = False


class SimpleSequenceTypeDetector:
    """Simplified sequence type detector"""
    
    @staticmethod
    def detect_label(description: str) -> Optional[int]:
        """Detect label from description"""
        if not description:
            return None
        
        desc_lower = description.lower()

        # Record-ID prefixes used by the released benchmark (pos_0000123 / neg_0000456)
        head = desc_lower.split('|')[0].strip()
        if head.startswith(('pos_', 'pos-', 'pos.', 'positive_')):
            return 1
        if head.startswith(('neg_', 'neg-', 'neg.', 'negative_')):
            return 0

        # Check for pipe separator
        if '|' in desc_lower:
            parts = desc_lower.split('|')
            for part in parts:
                part = part.strip()
                if part.startswith('label:'):
                    label_text = part[6:].strip()
                    if label_text in ['pathogen', 'positive', '1']:
                        return 1
                    elif label_text in ['non-pathogen', 'negative', '0']:
                        return 0
        
        # Check pattern at end of description
        desc_words = desc_lower.split()
        if len(desc_words) > 0:
            last_word = desc_words[-1].strip('|').strip(':').strip()
            if last_word in ['pathogen', 'positive', '1']:
                return 1
            elif last_word in ['non-pathogen', 'negative', '0']:
                return 0
        
        # Check markers in ID
        if '_negative' in desc_lower or '_neg_' in desc_lower:
            return 0
        elif '_positive' in desc_lower or '_pos_' in desc_lower:
            return 1
        
        # Hard-coded rules
        if 'non-pathogen' in desc_lower or 'nonpathogen' in desc_lower:
            return 0
        elif 'pathogen' in desc_lower and 'non-pathogen' not in desc_lower:
            return 1
        
        return None


class SimpleNTv3Predictor:
    """Simplified NTv3 model predictor with adjustable threshold"""
    
    def __init__(self, model_path: str, output_dir: str = None, threshold: float = 0.5,
                 ntv3_model: str = None):
        """
        Initialize predictor
        
        Args:
            model_path: Model file path
            output_dir: Output directory
            threshold: Classification threshold (default: 0.5)
            ntv3_model: folder holding the NT-V3-Pre-8M checkpoint; overrides the
                absolute path stored inside a checkpoint from a training host
        """
        if not IMPORT_SUCCESS:
            raise ImportError("Cannot import required modules")
        
        self.device = torch.device(str(config.device))
        self.model_path = model_path
        self.threshold = threshold
        self.ntv3_model = ntv3_model or os.environ.get('DUETSCREEN_NTV3_MODEL')
        
        # Validate threshold
        if not 0.0 <= threshold <= 1.0:
            print(f"⚠️  Warning: Threshold {threshold} is not between 0 and 1. Using default 0.5")
            self.threshold = 0.5
        
        # Set output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = os.path.basename(model_path).replace('.pth', '')
            output_dir = os.path.join(config.output_dir, f"prediction_{model_name}_threshold_{threshold}_{timestamp}")
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set unified matplotlib style
        plt.style.use('default')
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        print(f"🔍 Initializing NTv3 Predictor")
        print(f"  Model: {model_path}")
        print(f"  Output Directory: {output_dir}")
        print(f"  Classification Threshold: {threshold}")
        print(f"  Device: {self.device}")
        
        # Load model
        self.model, self.model_config = self._load_model()
        
        # Initialize type detector
        self.type_detector = SimpleSequenceTypeDetector()
        
        print(f"✅ Model loaded successfully")
    
    def _load_model(self):
        """Load model"""
        print(f"📂 Loading model: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')

        _seed = (checkpoint.get('config') or {}).get('seed')
        if _seed is not None:
            print(f"  Training seed recorded in checkpoint: {_seed}")
        
        # Get model configuration
        model_config = checkpoint.get('model_config', {})
        if not model_config:
            # Extract model config from training config
            train_config = checkpoint.get('config', {})
            if train_config:
                model_config = {
                    'transformer_model_repo': train_config.get('transformer_model_repo', 'InstaDeepAI/NTv3_8M_pre'),
                    'embedding_dim': train_config.get('embedding_dim', 256),
                    'd_model': train_config.get('d_model', 256),
                    'n_layer': train_config.get('n_layer', 2),
                    'projection_dim': train_config.get('projection_dim', 128),
                    'num_classes': train_config.get('num_classes', 1),
                    'variant_specialization_weight': train_config.get('variant_specialization_weight', 0.2),
                    'block_type': train_config.get('block_type', 'dual_path'),
                    'dropout_rate': train_config.get('dropout_rate', 0.2),
                    'use_path_selection': train_config.get('use_path_selection', True),
                    'path_selection_weight': train_config.get('path_selection_weight', 0.1),
                    'max_seq_len': train_config.get('max_seq_len', 2048),
                    'freeze_transformer': train_config.get('freeze_transformer', False),
                    'use_caching': train_config.get('use_caching', False),
                    'trust_remote_code': train_config.get('trust_remote_code', True),
                    'use_local_global_attn': train_config.get('use_local_global_attn', True),
                    'use_global_invariance': train_config.get('use_global_invariance', True),
                    'attn_num_heads': train_config.get('attn_num_heads', 4),
                    'use_flash_attention': train_config.get('use_flash_attention', True)
                }
            else:
                # Use default configuration
                model_config = {
                    'transformer_model_repo': 'InstaDeepAI/NTv3_8M_pre',
                    'embedding_dim': 256,
                    'd_model': 256,
                    'n_layer': 2,
                    'projection_dim': 128,
                    'num_classes': 1,
                    'variant_specialization_weight': 0.2,
                    'block_type': 'dual_path',
                    'dropout_rate': 0.2,
                    'use_path_selection': True,
                    'path_selection_weight': 0.1,
                    'max_seq_len': 2048,
                    'freeze_transformer': False,
                    'use_caching': False,
                    'trust_remote_code': True
                }
        
        # Resolve the encoder folder: explicit CLI/env override first, then the
        # path stored in the checkpoint only if it still exists locally.
        override = getattr(self, 'ntv3_model', None)
        stored = model_config.get('transformer_model_repo', '')
        if override:
            model_config['transformer_model_repo'] = override
        elif os.path.isabs(stored) and not os.path.isdir(stored):
            fallback = os.environ.get('DUETSCREEN_NTV3_MODEL')
            if fallback and os.path.isdir(fallback):
                model_config['transformer_model_repo'] = fallback
            else:
                raise FileNotFoundError(
                    'The NT-V3 checkpoint folder stored in this checkpoint does not exist on '
                    f'this machine ({stored}). Pass --ntv3-model /path/to/NTv3_8M_pre or set '
                    'DUETSCREEN_NTV3_MODEL.')

        print(f"📋 Model Configuration:")
        print(f"  Transformer Model: {model_config.get('transformer_model_repo', 'N/A')}")
        print(f"  Embedding Dimension: {model_config.get('embedding_dim', 'N/A')}")
        print(f"  Model Dimension: {model_config.get('d_model', 'N/A')}")
        print(f"  Number of Layers: {model_config.get('n_layer', 'N/A')}")
        print(f"  Block Type: {model_config.get('block_type', 'N/A')}")
        print(f"  Sequence Length: {model_config.get('max_seq_len', 'N/A')}")
        
        # Create model. Filter model_config down to parameters the model
        # constructor actually accepts: checkpoints store trainer-side flags
        # (pure_task_mode, ...) next to architecture selectors, and an unknown
        # key would throw and silently fall into the simplified rebuild below,
        # which loses head/architecture choices (e.g. classifier_head) and
        # leaves the restored head at random initialization.
        import inspect as _inspect
        from .model import NucleotideMambaModel as _ModelClass
        _accepted = set(_inspect.signature(_ModelClass.__init__).parameters)
        create_config = {k: v for k, v in model_config.items() if k in _accepted}
        # Every trained arm tokenizes at 2048; without this the rebuild would
        # silently fall back to create()'s 512 default and truncate all eval
        # inputs to a quarter length (2026-09-20 eval-pipeline fix). Old
        # checkpoints do not store max_seq_len, so default it here.
        create_config.setdefault('max_seq_len', 2048)

        # Checkpoints exported during development carry the internal run id in
        # block_type (e.g. 'v6:<internal-id>'). This package ships exactly one
        # released architecture, so any v6 marker is resolved to it.
        _bt = create_config.get('block_type', '')
        if isinstance(_bt, str) and _bt.startswith('v6:'):
            _cid = _bt.split(':')[1]
            if _cid != 'DuetScreen':
                create_config['block_type'] = 'v6:DuetScreen'
                print(f'  note: checkpoint block_type {_bt!r} mapped to the released model')

        # Create model
        try:
            model = create_nucleotide_mamba_model(**create_config)
        except Exception as e:
            print(f"❌ Error creating model: {str(e)}")
            # Try with simplified configuration
            simplified_config = {
                'transformer_model_repo': model_config.get('transformer_model_repo', 'InstaDeepAI/NTv3_8M_pre'),
                'embedding_dim': model_config.get('embedding_dim', 256),
                'd_model': model_config.get('d_model', 256),
                'n_layer': model_config.get('n_layer', 2),
                'projection_dim': model_config.get('projection_dim', 128),
                'num_classes': model_config.get('num_classes', 1),
                'block_type': model_config.get('block_type', 'dual_path'),
                'classifier_head': model_config.get('classifier_head', 'mlp'),
                'use_path_selection': model_config.get('use_path_selection', True),
                'max_seq_len': model_config.get('max_seq_len', 2048),
                'freeze_transformer': False
            }
            model = create_nucleotide_mamba_model(**simplified_config)
        
        # Load state dictionary
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle state dictionary mismatch
        model_state_dict = model.state_dict()
        
        try:
            # First try direct loading
            model.load_state_dict(state_dict)
            print("✅ Model loaded successfully with exact match")
        except RuntimeError as e:
            print(f"⚠️  Exact loading failed, attempting partial load: {str(e)[:200]}...")
            
            # Partial loading: only load matching keys
            matched_keys = []
            mismatched_keys = []
            new_state_dict = {}
            
            for key in model_state_dict.keys():
                if key in state_dict:
                    if model_state_dict[key].shape == state_dict[key].shape:
                        new_state_dict[key] = state_dict[key]
                        matched_keys.append(key)
                    else:
                        print(f"⚠️  Shape mismatch for {key}: model {model_state_dict[key].shape}, checkpoint {state_dict[key].shape}")
                        mismatched_keys.append(key)
                        new_state_dict[key] = model_state_dict[key]
                else:
                    mismatched_keys.append(key)
                    new_state_dict[key] = model_state_dict[key]
            
            model.load_state_dict(new_state_dict)
            print(f"📊 Partial loading: {len(matched_keys)} matched, {len(mismatched_keys)} mismatched")
        
        model.to(self.device)
        model.eval()
        
        print(f"✅ NTv3 Model Loaded Successfully")
        
        return model, model_config
    
    def parse_fasta_file(self, fasta_file: str) -> Tuple[List[str], List[str], List[Optional[int]]]:
        """
        Parse FASTA file
        
        Args:
            fasta_file: FASTA file path
            
        Returns:
            sequence_ids: Sequence ID list
            sequences: Sequence list
            labels: Label list (if available in file)
        """
        print(f"📄 Parsing FASTA File: {fasta_file}")
        
        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"FASTA file not found: {fasta_file}")
        
        sequence_ids = []
        sequences = []
        labels = []
        
        with open(fasta_file, 'r') as f:
            for record in SeqIO.parse(f, "fasta"):
                seq_id = record.id
                sequence = str(record.seq).upper()
                description = record.description
                
                # Check if sequence contains only valid characters
                valid_bases = set('ATCGN')
                if not all(base in valid_bases for base in sequence):
                    print(f"⚠️  Sequence {seq_id} contains invalid characters, skipping")
                    continue
                
                # Detect label
                label = self.type_detector.detect_label(description)
                
                sequence_ids.append(seq_id)
                sequences.append(sequence)
                labels.append(label)
        
        # Count label distribution
        label_counts = {
            'positive': sum(1 for label in labels if label == 1),
            'negative': sum(1 for label in labels if label == 0),
            'unknown': sum(1 for label in labels if label is None)
        }
        
        print(f"✅ FASTA File Parsing Completed")
        print(f"  Total Sequences: {len(sequences)}")
        print(f"  Positive Sequences (Pathogenic): {label_counts['positive']}")
        print(f"  Negative Sequences (Non-pathogenic): {label_counts['negative']}")
        print(f"  Unknown Label Sequences: {label_counts['unknown']}")
        
        return sequence_ids, sequences, labels
    
    def predict_batch(self, sequences: List[str], batch_size: int = 8, threshold: float = None) -> Tuple[List[float], List[int]]:
        """
        Run batch predictions using model
        
        Args:
            sequences: Sequence list
            batch_size: Batch size
            threshold: Classification threshold (optional, uses instance threshold if None)
            
        Returns:
            probabilities: Prediction probability list
            predictions: Predicted label list
        """
        if threshold is None:
            threshold = self.threshold
        
        print(f"\n🎯 Running Model Predictions (Threshold: {threshold})...")
        
        if not sequences:
            print("❌ No valid data for prediction")
            return [], []
        
        print(f"  Valid Data Items: {len(sequences)}")
        
        # Prepare data
        probabilities = [0.0] * len(sequences)
        predictions = [0] * len(sequences)
        
        # Batch prediction
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Predicting"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(sequences))
                
                batch_seqs = sequences[start_idx:end_idx]
                
                try:
                    # Forward pass
                    outputs = self.model.forward_sequence(
                        sequences=batch_seqs,
                        variant_type=None,
                        training_mode=False
                    )
                    
                    # Get prediction probabilities
                    class_pred = outputs[0]
                    batch_probs = torch.sigmoid(class_pred).cpu().numpy().flatten()
                    
                    # Apply custom threshold
                    batch_preds = (batch_probs > threshold).astype(int)
                    
                    # Save results
                    for i in range(len(batch_seqs)):
                        global_idx = start_idx + i
                        probabilities[global_idx] = float(batch_probs[i])
                        predictions[global_idx] = int(batch_preds[i])
                        
                except Exception as e:
                    print(f"⚠️  Error predicting batch {batch_idx}: {str(e)}")
                    # Add default values for failed batch
                    for i in range(len(batch_seqs)):
                        global_idx = start_idx + i
                        probabilities[global_idx] = 0.5
                        predictions[global_idx] = 0
        
        print(f"✅ Predictions Completed")
        
        return probabilities, predictions
    
    def calculate_metrics_at_thresholds(self, true_labels: List[Optional[int]], probabilities: List[float], 
                                      thresholds: List[float] = None) -> Dict[str, Any]:
        """
        Calculate metrics at different thresholds
        
        Args:
            true_labels: True label list
            probabilities: Prediction probability list
            thresholds: List of thresholds to evaluate
            
        Returns:
            Dictionary containing metrics at each threshold
        """
        # Filter labeled samples
        labeled_indices = [i for i, label in enumerate(true_labels) if label is not None]
        
        if not labeled_indices:
            print("⚠️  No labeled data, skipping threshold analysis")
            return None
        
        # Extract labeled data
        labeled_true = [true_labels[i] for i in labeled_indices]
        labeled_probs = [probabilities[i] for i in labeled_indices]
        
        true_np = np.array(labeled_true)
        probs_np = np.array(labeled_probs)
        
        # Default thresholds
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        threshold_metrics = {}
        
        print(f"\n📊 Calculating metrics at different thresholds...")
        print(f"  Labeled Samples: {len(labeled_true)}")
        print(f"  Thresholds: {thresholds}")
        
        for threshold in thresholds:
            # Apply threshold
            threshold_preds = (probs_np > threshold).astype(int)
            
            # Calculate metrics
            metrics = {}
            metrics['accuracy'] = accuracy_score(true_np, threshold_preds)
            metrics['f1_score'] = f1_score(true_np, threshold_preds, average='binary', zero_division=0)
            metrics['precision'] = precision_score(true_np, threshold_preds, zero_division=0)
            metrics['recall'] = recall_score(true_np, threshold_preds, zero_division=0)
            metrics['mcc'] = matthews_corrcoef(true_np, threshold_preds)
            
            # Confusion matrix
            cm = confusion_matrix(true_np, threshold_preds)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Class distribution
            metrics['predicted_positive'] = int(np.sum(threshold_preds == 1))
            metrics['predicted_negative'] = int(np.sum(threshold_preds == 0))
            
            threshold_metrics[threshold] = metrics
        
        # Find optimal threshold based on F1 score
        optimal_threshold = max(threshold_metrics.items(), key=lambda x: x[1]['f1_score'])[0]
        optimal_metrics = threshold_metrics[optimal_threshold]
        
        print(f"\n🎯 Optimal Threshold Analysis:")
        print(f"  Optimal Threshold (by F1 Score): {optimal_threshold:.2f}")
        print(f"  Optimal F1 Score: {optimal_metrics['f1_score']:.4f}")
        print(f"  Optimal Accuracy: {optimal_metrics['accuracy']:.4f}")
        print(f"  Optimal Precision: {optimal_metrics['precision']:.4f}")
        print(f"  Optimal Recall: {optimal_metrics['recall']:.4f}")
        
        return {
            'threshold_metrics': threshold_metrics,
            'optimal_threshold': optimal_threshold,
            'optimal_metrics': optimal_metrics
        }
    
    def calculate_metrics(self, true_labels: List[Optional[int]], predictions: List[int], 
                         probabilities: List[float], threshold: float = None) -> Dict[str, Any]:
        """
        Calculate evaluation metrics
        
        Args:
            true_labels: True label list
            predictions: Predicted label list
            probabilities: Prediction probability list
            threshold: Classification threshold used
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if threshold is None:
            threshold = self.threshold
        
        # Filter labeled samples
        labeled_indices = [i for i, label in enumerate(true_labels) if label is not None]
        
        if not labeled_indices:
            print("⚠️  No labeled data, skipping metric calculation")
            return None
        
        print(f"\n📊 Calculating evaluation metrics (Threshold: {threshold})...")
        print(f"  Labeled Samples: {len(labeled_indices)}")
        
        # Extract labeled data
        labeled_true = [true_labels[i] for i in labeled_indices]
        labeled_pred = [predictions[i] for i in labeled_indices]
        labeled_probs = [probabilities[i] for i in labeled_indices]
        
        true_np = np.array(labeled_true)
        pred_np = np.array(labeled_pred)
        probs_np = np.array(labeled_probs)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(true_np, pred_np)
        metrics['f1_score'] = f1_score(true_np, pred_np, average='binary', zero_division=0)
        metrics['precision'] = precision_score(true_np, pred_np, zero_division=0)
        metrics['recall'] = recall_score(true_np, pred_np, zero_division=0)
        metrics['mcc'] = matthews_corrcoef(true_np, pred_np)
        
        # AUC and ROC
        if len(np.unique(true_np)) > 1:
            try:
                metrics['auc'] = roc_auc_score(true_np, probs_np)
                # Calculate ROC curve
                fpr, tpr, roc_thresholds = roc_curve(true_np, probs_np)
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': roc_thresholds.tolist()
                }
                
                # Calculate precision-recall curve
                precision_curve, recall_curve, pr_thresholds = precision_recall_curve(true_np, probs_np)
                metrics['pr_curve'] = {
                    'precision': precision_curve.tolist(),
                    'recall': recall_curve.tolist(),
                    'thresholds': pr_thresholds.tolist()
                }
                
                # Calculate AUPRC (Area Under Precision-Recall Curve)
                metrics['auprc'] = average_precision_score(true_np, probs_np)
                
            except:
                metrics['auc'] = 0.0
                metrics['auprc'] = 0.0
                metrics['roc_curve'] = None
                metrics['pr_curve'] = None
        else:
            metrics['auc'] = 0.0
            metrics['auprc'] = 0.0
            metrics['roc_curve'] = None
            metrics['pr_curve'] = None
        
        # Brier Score
        if len(probs_np) > 0:
            brier_score = np.mean((probs_np - true_np) ** 2)
            metrics['brier_score'] = float(brier_score)
        else:
            metrics['brier_score'] = 1.0
        
        # Confusion matrix
        cm = confusion_matrix(true_np, pred_np)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(
            true_np, pred_np, 
            target_names=['Non-pathogenic', 'Pathogenic'],
            output_dict=True,
            zero_division=0
        )
        
        # Class distribution
        metrics['class_distribution'] = {
            'negative': int(np.sum(true_np == 0)),
            'positive': int(np.sum(true_np == 1)),
            'total': len(true_np)
        }
        
        # Prediction distribution
        metrics['prediction_distribution'] = {
            'predicted_negative': int(np.sum(pred_np == 0)),
            'predicted_positive': int(np.sum(pred_np == 1)),
            'total': len(pred_np)
        }
        
        # Store threshold information
        metrics['threshold_used'] = threshold
        
        print(f"✅ Metrics calculation completed")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  MCC: {metrics['mcc']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  AUPRC: {metrics.get('auprc', 0.0):.4f}")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")
        
        return metrics
    
    def create_visualizations(self, true_labels: List[Optional[int]], predictions: List[int], 
                            probabilities: List[float], sequence_ids: List[str], threshold: float = None):
        """
        Create visualization charts
        
        Args:
            true_labels: True label list
            predictions: Predicted label list
            probabilities: Prediction probability list
            sequence_ids: Sequence ID list
            threshold: Classification threshold used
        """
        if threshold is None:
            threshold = self.threshold
        
        print(f"\n🎨 Creating visualization charts...")
        
        # Filter labeled samples
        labeled_indices = [i for i, label in enumerate(true_labels) if label is not None]
        
        if not labeled_indices:
            print("⚠️  No labeled data, skipping visualization")
            return
        
        # Extract labeled data
        labeled_true = [true_labels[i] for i in labeled_indices]
        labeled_pred = [predictions[i] for i in labeled_indices]
        labeled_probs = [probabilities[i] for i in labeled_indices]
        labeled_ids = [sequence_ids[i] for i in labeled_indices]
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'Nucleotide Transformer v3 Prediction Results (Threshold: {threshold})', 
                    fontsize=18, fontweight='bold')
        
        # 1. Prediction Probability Distribution
        ax1 = axes[0, 0]
        
        # Group by true label
        pos_indices = [i for i, label in enumerate(labeled_true) if label == 1]
        neg_indices = [i for i, label in enumerate(labeled_true) if label == 0]
        
        if pos_indices:
            pos_probs = [labeled_probs[i] for i in pos_indices]
            ax1.hist(pos_probs, bins=20, alpha=0.7, label='Pathogenic', color='red', density=True)
        
        if neg_indices:
            neg_probs = [labeled_probs[i] for i in neg_indices]
            ax1.hist(neg_probs, bins=20, alpha=0.7, label='Non-pathogenic', color='blue', density=True)
        
        ax1.axvline(x=threshold, color='black', linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Decision Threshold: {threshold}')
        ax1.set_xlabel('Prediction Probability')
        ax1.set_ylabel('Density')
        ax1.set_title('Prediction Probability Distribution (by True Label)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Confusion Matrix
        ax2 = axes[0, 1]
        
        cm = confusion_matrix(labeled_true, labeled_pred)
        im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        # Add numerical labels
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['Non-pathogenic', 'Pathogenic'])
        ax2.set_yticklabels(['Non-pathogenic', 'Pathogenic'])
        ax2.set_title(f'Confusion Matrix (Threshold: {threshold})')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2)
        
        # 3. ROC Curve
        ax3 = axes[1, 0]
        
        if len(np.unique(labeled_true)) > 1:
            fpr, tpr, _ = roc_curve(labeled_true, labeled_probs)
            auc_score = roc_auc_score(labeled_true, labeled_probs)
            
            ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
            ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax3.legend(loc="lower right")
            ax3.grid(True, alpha=0.3)
            
            # Mark threshold point
            if len(labeled_probs) > 0:
                # Find closest threshold point on ROC curve
                threshold_idx = np.argmin(np.abs(np.array(labeled_probs) - threshold))
                if threshold_idx < len(fpr) and threshold_idx < len(tpr):
                    ax3.scatter(fpr[threshold_idx], tpr[threshold_idx], 
                              color='red', s=100, zorder=5, label=f'Threshold: {threshold}')
                    ax3.legend(loc="lower right")
        else:
            ax3.text(0.5, 0.5, "ROC curve requires both classes", 
                    ha='center', va='center', fontsize=12)
            ax3.set_title('ROC Curve (Not enough classes)')
        
        # 4. Precision-Recall Curve
        ax4 = axes[1, 1]
        
        if len(np.unique(labeled_true)) > 1:
            precision, recall, _ = precision_recall_curve(labeled_true, labeled_probs)
            # Calculate AUPRC
            auprc = average_precision_score(labeled_true, labeled_probs)
            
            ax4.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUPRC = {auprc:.3f})')
            ax4.set_xlabel('Recall')
            ax4.set_ylabel('Precision')
            ax4.set_title('Precision-Recall Curve')
            ax4.legend(loc="lower left")
            ax4.grid(True, alpha=0.3)
            
            # Mark threshold point
            if len(labeled_probs) > 0:
                # Find closest threshold point
                threshold_idx = np.argmin(np.abs(np.array(labeled_probs) - threshold))
                if threshold_idx < len(precision) and threshold_idx < len(recall):
                    ax4.scatter(recall[threshold_idx], precision[threshold_idx], 
                              color='red', s=100, zorder=5, label=f'Threshold: {threshold}')
                    ax4.legend(loc="lower left")
        else:
            ax4.text(0.5, 0.5, "PR curve requires both classes", 
                    ha='center', va='center', fontsize=12)
            ax4.set_title('Precision-Recall Curve (Not enough classes)')
        
        # 5. Prediction Probability Scatter Plot
        ax5 = axes[2, 0]
        
        # Color by correctness
        correct_indices = [i for i in range(len(labeled_true)) if labeled_true[i] == labeled_pred[i]]
        incorrect_indices = [i for i in range(len(labeled_true)) if labeled_true[i] != labeled_pred[i]]
        
        if correct_indices:
            correct_probs = [labeled_probs[i] for i in correct_indices]
            ax5.scatter(range(len(correct_probs)), correct_probs, 
                      label='Correct Predictions', alpha=0.7, s=30, color='green')
        
        if incorrect_indices:
            incorrect_probs = [labeled_probs[i] for i in incorrect_indices]
            start_idx = len(correct_probs) if correct_indices else 0
            ax5.scatter(range(start_idx, start_idx + len(incorrect_probs)), 
                      incorrect_probs, label='Incorrect Predictions', alpha=0.7, s=30, color='red')
        
        ax5.axhline(y=threshold, color='black', linestyle='--', linewidth=2, 
                   alpha=0.8, label=f'Threshold: {threshold}')
        ax5.set_xlabel('Sample Index')
        ax5.set_ylabel('Prediction Probability')
        ax5.set_title('Prediction Probability Distribution (by Correctness)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Threshold Analysis
        ax6 = axes[2, 1]
        
        # Calculate metrics at different thresholds
        thresholds = np.linspace(0.1, 0.9, 9)
        f1_scores = []
        accuracies = []
        
        for t in thresholds:
            t_preds = [1 if prob > t else 0 for prob in labeled_probs]
            f1_scores.append(f1_score(labeled_true, t_preds, average='binary', zero_division=0))
            accuracies.append(accuracy_score(labeled_true, t_preds))
        
        ax6.plot(thresholds, f1_scores, 'o-', linewidth=2, markersize=8, label='F1 Score')
        ax6.plot(thresholds, accuracies, 's-', linewidth=2, markersize=8, label='Accuracy')
        
        # Mark current threshold
        current_f1 = f1_score(labeled_true, labeled_pred, average='binary', zero_division=0)
        current_acc = accuracy_score(labeled_true, labeled_pred)
        
        ax6.scatter(threshold, current_f1, color='red', s=150, zorder=5, 
                   label=f'Current Threshold (F1: {current_f1:.3f})')
        ax6.scatter(threshold, current_acc, color='blue', s=150, zorder=5, 
                   label=f'Current Threshold (Acc: {current_acc:.3f})')
        
        ax6.set_xlabel('Decision Threshold')
        ax6.set_ylabel('Score')
        ax6.set_title('Performance at Different Thresholds')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        plot_path = os.path.join(self.output_dir, f"prediction_visualization_threshold_{threshold}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization chart saved: {plot_path}")
        
        # Create additional threshold analysis chart
        self.create_threshold_analysis_chart(labeled_true, labeled_probs, threshold)
    
    def create_threshold_analysis_chart(self, true_labels: List[int], probabilities: List[float], 
                                       current_threshold: float):
        """Create detailed threshold analysis chart"""
        if len(np.unique(true_labels)) <= 1:
            return
        
        # Calculate metrics at many thresholds
        thresholds = np.linspace(0.01, 0.99, 99)
        f1_scores = []
        precisions = []
        recalls = []
        accuracies = []
        
        true_np = np.array(true_labels)
        probs_np = np.array(probabilities)
        
        for t in thresholds:
            preds = (probs_np > t).astype(int)
            f1_scores.append(f1_score(true_np, preds, average='binary', zero_division=0))
            precisions.append(precision_score(true_np, preds, zero_division=0))
            recalls.append(recall_score(true_np, preds, zero_division=0))
            accuracies.append(accuracy_score(true_np, preds))
        
        # Find optimal thresholds
        optimal_f1_idx = np.argmax(f1_scores)
        optimal_f1_threshold = thresholds[optimal_f1_idx]
        optimal_f1_score = f1_scores[optimal_f1_idx]
        
        optimal_acc_idx = np.argmax(accuracies)
        optimal_acc_threshold = thresholds[optimal_acc_idx]
        optimal_acc_score = accuracies[optimal_acc_idx]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Detailed Threshold Analysis (Current: {current_threshold:.2f})', 
                    fontsize=16, fontweight='bold')
        
        # 1. F1 Score vs Threshold
        ax1 = axes[0, 0]
        ax1.plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1 Score')
        ax1.axvline(x=current_threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Current: {current_threshold:.2f}')
        ax1.axvline(x=optimal_f1_threshold, color='green', linestyle='--', alpha=0.7, 
                   label=f'Optimal F1: {optimal_f1_threshold:.2f}')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('F1 Score')
        ax1.set_title(f'F1 Score vs Threshold (Optimal: {optimal_f1_threshold:.2f}, F1: {optimal_f1_score:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision-Recall vs Threshold
        ax2 = axes[0, 1]
        ax2.plot(thresholds, precisions, 'g-', linewidth=2, label='Precision')
        ax2.plot(thresholds, recalls, 'r-', linewidth=2, label='Recall')
        ax2.axvline(x=current_threshold, color='black', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision and Recall vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Accuracy vs Threshold
        ax3 = axes[1, 0]
        ax3.plot(thresholds, accuracies, 'purple', linewidth=2, label='Accuracy')
        ax3.axvline(x=current_threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Current: {current_threshold:.2f}')
        ax3.axvline(x=optimal_acc_threshold, color='green', linestyle='--', alpha=0.7, 
                   label=f'Optimal Acc: {optimal_acc_threshold:.2f}')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Accuracy')
        ax3.set_title(f'Accuracy vs Threshold (Optimal: {optimal_acc_threshold:.2f}, Acc: {optimal_acc_score:.3f})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Combined view
        ax4 = axes[1, 1]
        ax4.plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1 Score')
        ax4.plot(thresholds, accuracies, 'purple', linewidth=2, label='Accuracy')
        ax4.plot(thresholds, precisions, 'g-', linewidth=1.5, label='Precision', alpha=0.7)
        ax4.plot(thresholds, recalls, 'r-', linewidth=1.5, label='Recall', alpha=0.7)
        ax4.axvline(x=current_threshold, color='black', linestyle='--', linewidth=2, 
                   alpha=0.8, label=f'Current: {current_threshold:.2f}')
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Score')
        ax4.set_title('All Metrics vs Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        analysis_path = os.path.join(self.output_dir, f"detailed_threshold_analysis.png")
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Detailed threshold analysis saved: {analysis_path}")
        
        # Save threshold analysis data
        threshold_data = {
            'thresholds': thresholds.tolist(),
            'f1_scores': f1_scores,
            'precisions': precisions,
            'recalls': recalls,
            'accuracies': accuracies,
            'current_threshold': current_threshold,
            'optimal_f1_threshold': float(optimal_f1_threshold),
            'optimal_f1_score': float(optimal_f1_score),
            'optimal_acc_threshold': float(optimal_acc_threshold),
            'optimal_acc_score': float(optimal_acc_score)
        }
        
        data_path = os.path.join(self.output_dir, "threshold_analysis_data.json")
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(threshold_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Threshold analysis data saved: {data_path}")
    
    def save_results(self, sequence_ids: List[str], sequences: List[str], 
                    true_labels: List[Optional[int]], predictions: List[int], 
                    probabilities: List[float], metrics: dict = None, threshold: float = None):
        """
        Save prediction results
        
        Args:
            sequence_ids: Sequence ID list
            sequences: Sequence list
            true_labels: True label list
            predictions: Predicted label list
            probabilities: Prediction probability list
            metrics: Evaluation metrics
            threshold: Classification threshold used
        """
        if threshold is None:
            threshold = self.threshold
        
        print(f"\n💾 Saving prediction results...")
        
        # 1. Save as CSV format
        csv_file = os.path.join(self.output_dir, f"predictions_threshold_{threshold}.csv")
        
        data = []
        for i, (seq_id, seq, label, pred, prob) in enumerate(zip(
            sequence_ids, sequences, true_labels, predictions, probabilities)):
            
            row = {
                'sequence_id': seq_id,
                'sequence_length': len(seq),
                'true_label': 'Unknown' if label is None else ('Pathogenic' if label == 1 else 'Non-pathogenic'),
                'predicted_label': 'Pathogenic' if pred == 1 else 'Non-pathogenic',
                'prediction_probability': prob,
                'prediction_confidence': 'High' if abs(prob - threshold) > 0.3 else ('Medium' if abs(prob - threshold) > 0.1 else 'Low'),
                'distance_to_threshold': abs(prob - threshold)
            }
            
            if label is not None:
                row['is_correct'] = 'Yes' if label == pred else 'No'
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"✅ Predictions saved to CSV: {csv_file}")
        
        # 2. Save as JSON format
        json_file = os.path.join(self.output_dir, "detailed_results.json")
        
        detailed_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_path': self.model_path,
            'model_config': self.model_config,
            'classification_threshold': threshold,
            'total_sequences': len(sequence_ids),
            'predictions': []
        }
        
        for i in range(len(sequence_ids)):
            pred_info = {
                'sequence_id': sequence_ids[i],
                'sequence_length': len(sequences[i]),
                'true_label': true_labels[i] if true_labels[i] is not None else None,
                'predicted_label': int(predictions[i]),
                'prediction_probability': float(probabilities[i]),
                'distance_to_threshold': float(abs(probabilities[i] - threshold))
            }
            detailed_data['predictions'].append(pred_info)
        
        if metrics:
            detailed_data['metrics'] = metrics
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Detailed results saved to JSON: {json_file}")
        
        # 3. Save metrics report
        if metrics:
            report_file = os.path.join(self.output_dir, f"prediction_report_threshold_{threshold}.txt")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("                 NTv3 PREDICTION RESULTS REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Prediction Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model Path: {self.model_path}\n")
                f.write(f"Classification Threshold: {threshold}\n")
                f.write(f"Model Configuration: {self.model_config.get('transformer_model_repo', 'N/A')}\n")
                f.write(f"Output Directory: {self.output_dir}\n\n")
                
                f.write("📊 DATA STATISTICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Sequences: {len(sequence_ids)}\n")
                f.write(f"Labeled Sequences: {sum(1 for label in true_labels if label is not None)}\n")
                f.write(f"Unlabeled Sequences: {sum(1 for label in true_labels if label is None)}\n\n")
                
                f.write("🎯 PREDICTION SUMMARY:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Predicted Pathogenic: {sum(predictions)}\n")
                f.write(f"Predicted Non-pathogenic: {len(predictions) - sum(predictions)}\n")
                f.write(f"Average Prediction Probability: {np.mean(probabilities):.4f}\n")
                f.write(f"Median Prediction Probability: {np.median(probabilities):.4f}\n")
                f.write(f"Standard Deviation: {np.std(probabilities):.4f}\n\n")
                
                f.write("📈 PERFORMANCE METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall: {metrics['recall']:.4f}\n")
                f.write(f"MCC: {metrics['mcc']:.4f}\n")
                f.write(f"AUC: {metrics['auc']:.4f}\n")
                if 'auprc' in metrics:
                    f.write(f"AUPRC: {metrics['auprc']:.4f}\n")
                f.write(f"Brier Score: {metrics['brier_score']:.4f}\n\n")
                
                if 'confusion_matrix' in metrics:
                    f.write("📊 CONFUSION MATRIX:\n")
                    f.write("-" * 40 + "\n")
                    cm = metrics['confusion_matrix']
                    f.write(f"              Predicted Non-path    Predicted Path\n")
                    f.write(f"True Non-path        {cm[0][0]:8d}            {cm[0][1]:8d}\n")
                    f.write(f"True Path            {cm[1][0]:8d}            {cm[1][1]:8d}\n\n")
                
                f.write("📁 OUTPUT FILES:\n")
                f.write("-" * 40 + "\n")
                f.write(f"1. predictions_threshold_{threshold}.csv - CSV predictions\n")
                f.write(f"2. detailed_results.json - JSON detailed results\n")
                f.write(f"3. prediction_report_threshold_{threshold}.txt - This report\n")
                f.write(f"4. prediction_visualization_threshold_{threshold}.png - Visualization charts\n")
                f.write(f"5. detailed_threshold_analysis.png - Threshold analysis\n")
                f.write(f"6. threshold_analysis_data.json - Threshold analysis data\n")
                
                f.write("\n" + "=" * 80 + "\n")
            
            print(f"✅ Prediction report saved: {report_file}")
        
        return csv_file, json_file
    
    def predict(self, fasta_file: str, batch_size: int = 8, threshold: float = None):
        """
        Predict FASTA file
        
        Args:
            fasta_file: FASTA file path
            batch_size: Batch size
            threshold: Classification threshold (overrides instance threshold if provided)
            
        Returns:
            probabilities, predictions, metrics
        """
        if threshold is not None:
            # Use provided threshold
            prediction_threshold = threshold
            print(f"📢 Using provided threshold: {prediction_threshold}")
        else:
            # Use instance threshold
            prediction_threshold = self.threshold
        
        print("=" * 70)
        print(f"🔍 NTv3 FASTA PREDICTION")
        print(f"  Model: {os.path.basename(self.model_path)}")
        print(f"  FASTA File: {fasta_file}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Classification Threshold: {prediction_threshold}")
        print("=" * 70)
        
        try:
            # 1. Parse FASTA file
            sequence_ids, sequences, labels = self.parse_fasta_file(fasta_file)
            
            if not sequences:
                print("❌ No valid sequences in FASTA file")
                return None
            
            # 2. Run predictions with custom threshold
            probabilities, predictions = self.predict_batch(
                sequences, 
                batch_size, 
                threshold=prediction_threshold
            )
            
            if not probabilities:
                print("❌ Prediction failed")
                return None
            
            # 3. Calculate metrics
            metrics = None
            if any(label is not None for label in labels):
                metrics = self.calculate_metrics(labels, predictions, probabilities, prediction_threshold)
                
                # Additional threshold analysis
                threshold_analysis = self.calculate_metrics_at_thresholds(labels, probabilities)
                if threshold_analysis:
                    if metrics:
                        metrics['threshold_analysis'] = threshold_analysis
            
            # 4. Create visualization charts
            self.create_visualizations(labels, predictions, probabilities, sequence_ids, prediction_threshold)
            
            # 5. Save results
            self.save_results(sequence_ids, sequences, labels, predictions, probabilities, metrics, prediction_threshold)
            
            print(f"\n{'='*70}")
            print(f"🎉 PREDICTION COMPLETED!")
            print(f"{'='*70}")
            
            # Print summary
            print(f"\n📊 PREDICTION SUMMARY:")
            print(f"  Total Sequences: {len(sequences)}")
            print(f"  Predicted Pathogenic: {sum(predictions)}")
            print(f"  Predicted Non-pathogenic: {len(predictions) - sum(predictions)}")
            print(f"  Average Prediction Probability: {np.mean(probabilities):.4f}")
            print(f"  Classification Threshold: {prediction_threshold}")
            
            if metrics:
                print(f"\n📈 PERFORMANCE METRICS:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1 Score: {metrics['f1_score']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  AUC: {metrics['auc']:.4f}")
                if 'auprc' in metrics:
                    print(f"  AUPRC: {metrics['auprc']:.4f}")
                print(f"  Brier Score: {metrics['brier_score']:.4f}")
                
                # Print optimal threshold info if available
                if 'threshold_analysis' in metrics:
                    optimal = metrics['threshold_analysis']['optimal_threshold']
                    optimal_f1 = metrics['threshold_analysis']['optimal_metrics']['f1_score']
                    print(f"\n🎯 THRESHOLD OPTIMIZATION:")
                    print(f"  Optimal Threshold (by F1): {optimal:.3f}")
                    print(f"  Optimal F1 Score: {optimal_f1:.3f}")
                    print(f"  Current F1 Score: {metrics['f1_score']:.3f}")
                    print(f"  Improvement Potential: {optimal_f1 - metrics['f1_score']:.3f}")
            
            print(f"\n📁 Results saved to directory: {self.output_dir}")
            
            return probabilities, predictions, metrics
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function"""
    if not IMPORT_SUCCESS:
        print("❌ Cannot import required modules, please check if files exist")
        return
    
    parser = argparse.ArgumentParser(description="Nucleotide Transformer v3 FASTA Prediction Tool with Adjustable Threshold")
    parser.add_argument('--ntv3-model', default=None,
                        help='local folder of the NT-V3-Pre-8M checkpoint')
    
    # Required parameters
    parser.add_argument("--fasta", type=str, required=True, help="Input FASTA file path")
    parser.add_argument("--model", type=str, required=True, help="Trained model path (.pth file)")
    
    # Optional parameters
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 8)")
    parser.add_argument("--threshold", type=float, default=0.5, 
                       help="Classification threshold (0.0-1.0, default: 0.5)")
    
    args = parser.parse_args()
    
    # Check file existence
    if not os.path.exists(args.fasta):
        print(f"❌ FASTA file not found: {args.fasta}")
        return
    
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print(f"❌ Invalid threshold: {args.threshold}. Must be between 0.0 and 1.0")
        return
    
    # Create predictor and run prediction
    try:
        predictor = SimpleNTv3Predictor(
            model_path=args.model,
            output_dir=args.output_dir,
            threshold=args.threshold,
            ntv3_model=getattr(args, 'ntv3_model', None),
        )
        
        predictor.predict(args.fasta, args.batch_size, args.threshold)
        
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
