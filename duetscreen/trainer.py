"""
Nucleotide Transformer v3 + Mamba Trainer
Fully trainable NTv3 model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import time
import numpy as np
from tqdm import tqdm

# Import custom modules
from .config import config
from .model import create_nucleotide_mamba_model
from .loss import TripletContrastiveLoss, HardTripletLoss
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from torch.amp import autocast, GradScaler
from .data import create_nucleotide_dataloader, variant_indices
from .reproducibility import seed_training
from .data_split import prepare_checked_split, verify_prepared_split


# Controls dataset membership independently of parameter initialization.
DATA_SPLIT_SEED = 42


def build_loss_weights(config):
    """Objective weights for the classification-family losses.

    Legacy ours: 0.8·BCE(orig) + 0.1·BCE(T1) + 0.1·BCE(T2) + 0.3·contrastive
    + 0.1·path, all divided by 1.4. Reweighted arms (control): the
    original-sample task keeps ``1 - aux_budget``; the auxiliary terms share
    ``aux_budget`` proportionally to their legacy weights, with no
    renormalization, so the primary task dominates the gradient.
    """
    legacy = {"orig": 0.8, "sem": 0.1, "conf": 0.1, "contrast": 0.3, "path": 0.1}
    if not getattr(config, "objective_reweight", False):
        weights = {**legacy, "norm": 1.4}
    else:
        budget = float(getattr(config, "aux_budget", 0.1))
        total_aux = sum(v for k, v in legacy.items() if k != "orig")
        weights = {k: budget * v / total_aux for k, v in legacy.items() if k != "orig"}
        weights["orig"] = 1.0 - budget
        weights["norm"] = 1.0
    # Ladder rung 5 (control): dual-path WITHOUT the contrastive term —
    # keep view BCE + path supervision + augmentation, zero only contrast.
    if getattr(config, "objective_no_contrastive", False):
        weights["contrast"] = 0.0
    # D_L control (control specification): dual-path with path KL + entropy
    # switched off; remaining terms keep their coefficients and the total
    # keeps its historical normalization (no rescaling).
    if getattr(config, "objective_no_path", False):
        weights["path"] = 0.0
    return weights


class NucleotideMambaTrainer:
    """Nucleotide Transformer v3 + Mamba Trainer"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        # Arm C (pure-task protocol): plain original-sample BCE only.
        self.pure_mode = bool(getattr(config, "pure_task_mode", False))
        if self.pure_mode:
            print("🧪 Pure-task protocol: original samples only, plain BCE "
                  "(no views / contrastive / path supervision)")
        # Objective weights (legacy 0.8/0.1/0.1/0.3/0.1 ÷1.4, or reweighted
        # 0.9 primary + 0.1 auxiliary budget when objective_reweight=True).
        self.loss_weights = build_loss_weights(config)
        if getattr(config, "objective_reweight", False):
            print(f"⚖️ Reweighted objective: orig={self.loss_weights['orig']:.2f} "
                  f"aux_budget={getattr(config, 'aux_budget', 0.1)} "
                  f"(sem={self.loss_weights['sem']:.4f} conf={self.loss_weights['conf']:.4f} "
                  f"contrast={self.loss_weights['contrast']:.4f} path={self.loss_weights['path']:.4f})")
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        print(f"🚀 Initializing fully trainable Nucleotide Transformer v3 + Mamba Trainer")
        print(f"  Device: {self.device}")
        print(f"  Output directory: {config.output_dir}")
        print(f"  Batch size: {config.batch_size} ⚠️ (NTv3 requires large memory)")
        print(f"  Using Nucleotide Transformer v3: {getattr(config, 'use_nucleotide_transformer', False)}")
    
    def prepare_data(self, variants_data):
        """Prepare data - Using Nucleotide Transformer v3"""
        print(f"\n📊 Preparing Nucleotide Transformer v3 training data...")
        
        # Calculate maximum sequence length (number of DNA bases)
        max_seq_len = getattr(self.config, 'max_seq_len', 512)
        
        # Check data
        if not variants_data:
            raise ValueError("No variant data provided")
        
        print(f"  Total samples: {len(variants_data):,}")

        if getattr(self.config, "use_prepared_split", False):
            # Membership is fixed by the delivered dataset: records already
            # carry a split tag and this pipeline never re-partitions them.
            train_data = [r for r in variants_data if r.get("split") == "train"]
            val_data = [r for r in variants_data if r.get("split") == "val"]
            stray = len(variants_data) - len(train_data) - len(val_data)
            if stray:
                raise ValueError(
                    f"{stray} variant records lack a 'train'/'val' split tag; "
                    "prepared-split mode requires tagged records")
            split_summary = verify_prepared_split(
                variants_data, self.config.independent_test_files)
        else:
            # Validate the actual input records, including every attached view.
            # The reserved subset is never passed to fitting or validation loaders.
            train_data, val_data, test_data, split_summary = prepare_checked_split(
                variants_data,
                self.config.independent_test_files,
                seed=DATA_SPLIT_SEED,
            )
            print(f"  Reserved internal test set: {len(test_data):,} samples (not evaluated)")
            del test_data

        self.data_split_summary = split_summary
        split_path = os.path.join(self.config.output_dir, "data_split.json")
        try:
            with open(split_path, 'x', encoding='utf-8') as stream:
                json.dump(split_summary, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            with open(split_path, encoding='utf-8') as stream:
                previous_split = json.load(stream)
            if previous_split != split_summary:
                raise ValueError("Existing data_split.json describes different inputs or membership; use a separate output directory.")

        print(f"📊 Data split:")
        print(f"  Training set: {len(train_data):,} samples")
        print(f"  Validation set: {len(val_data):,} samples")
        print(f"  Split manifest: {split_path}")
        
        # Create data loaders
        train_loader = create_nucleotide_dataloader(
            segments=train_data,
            batch_size=self.config.batch_size,
            max_seq_len=max_seq_len,
            shuffle=True,
            num_workers=min(4, os.cpu_count()),
            use_cache=config.use_caching,  # Real-time processing
            seed=self.config.seed,  # Shuffle/workers vary with the training run.
            pure_task=self.pure_mode,
        )
        
        val_loader = create_nucleotide_dataloader(
            segments=val_data,
            batch_size=self.config.batch_size,
            max_seq_len=max_seq_len,
            shuffle=False,
            num_workers=min(4, os.cpu_count()),
            use_cache=config.use_caching,
            # A separate fixed worker stream keeps validation sampling independent
            # of the training seed. It continues across epochs with persistent workers.
            seed=DATA_SPLIT_SEED,
            pure_task=self.pure_mode,
        )
        
        return train_loader, val_loader
    
    def create_model(self):
        """Create fully trainable Nucleotide Transformer v3 + Mamba model"""
        print(f"\n🧠 Creating fully trainable Nucleotide Transformer v3 + Mamba model...")
        
        # Get model parameters from config
        model_kwargs = {
            'transformer_model_repo': getattr(self.config, 'transformer_model_repo', 'InstaDeepAI/NTv3_8M_pre'),
            'embedding_dim': getattr(self.config, 'embedding_dim', 256),
            'd_model': getattr(self.config, 'd_model', 256),
            'n_layer': getattr(self.config, 'n_layer', 4),
            'projection_dim': getattr(self.config, 'projection_dim', 128),
            'num_classes': getattr(self.config, 'num_classes', 1),
            'variant_specialization_weight': getattr(self.config, 'variant_specialization_weight', 0.2),
            'block_type': getattr(self.config, 'block_type', 'dual_path'),
            'classifier_head': getattr(self.config, 'classifier_head', 'mlp'),
            'readout_pool': getattr(self.config, 'readout_pool', 'attention'),
            'readout_fusion': getattr(self.config, 'readout_fusion', 'linear'),
            'readout_rank': getattr(self.config, 'readout_rank', 64),
            'readout_source': getattr(self.config, 'readout_source', 'input'),
            'zero_internal_dropout': getattr(self.config, 'zero_internal_dropout', False),
            'dropout_rate': getattr(self.config, 'dropout_rate', 0.1),
            'use_path_selection': getattr(self.config, 'use_path_selection', True),
            'path_selection_weight': getattr(self.config, 'path_selection_weight', 0.1),
            'max_seq_len': getattr(self.config, 'max_seq_len', 512),
            'freeze_transformer': getattr(self.config, 'freeze_transformer', True),  # Fully trainable
            'use_caching': getattr(self.config, 'use_caching', True),
            'trust_remote_code': getattr(self.config, 'trust_remote_code', True),
            'use_local_global_attn': getattr(self.config, 'use_local_global_attn', True),
            'use_global_invariance': getattr(self.config, 'use_global_invariance', True),
            'attn_num_heads': getattr(self.config, 'attn_num_heads', 4),
            # 'num_segments': getattr(self.config, 'num_segments', 8),
            'use_flash_attention': getattr(self.config, 'use_flash_attention', True),
            'enable_tf32': getattr(self.config, 'enable_tf32', True),
            'compile_model': getattr(self.config, 'compile_model', False),
            'device': str(self.device)
        }
        
        model = create_nucleotide_mamba_model(**model_kwargs)
        
        # Print model statistics
        model_summary = model.get_model_summary()
        print(f"✅ Model created successfully")
        print(f"  Total parameters: {model_summary['total_parameters']:,}")
        print(f"  Trainable parameters: {model_summary['trainable_parameters']:,} ⚠️ (fully trainable)")
        print(f"  NTv3 parameters: {model_summary.get('ntv3_parameters', 0):,}")
        print(f"  NTv3 trainable: {model_summary.get('ntv3_trainable', 0):,}")
        print(f"  Mamba parameters: {model_summary.get('mamba_parameters', 0):,}")
        print(f"  Parameter size: {model_summary.get('parameter_mb', 0):.2f} MB")
        print(f"  Model dimension: {model_summary.get('d_model', 0)}")
        print(f"  Model architecture: {model_summary.get('model_architecture', '?')}")
        
        return model
    
    def create_loss_functions(self):
        """Create loss functions"""
        # Classification loss
        class_criterion = nn.BCEWithLogitsLoss()
        
        # Contrastive loss
        if getattr(self.config, 'use_hard_triplet', False):
            contrastive_criterion = HardTripletLoss(margin=getattr(self.config, 'triplet_margin', 1.0))
        else:
            contrastive_criterion = TripletContrastiveLoss(
                margin=getattr(self.config, 'triplet_margin', 1.0),
                temperature=getattr(self.config, 'triplet_temperature', 0.1)
            )
        
        return class_criterion, contrastive_criterion
    
    def create_optimizer(self, model):
        """Create optimizer - NTv3 fully trainable, need to optimize all parameters"""
        # Separate NTv3 parameters from others, can set different learning rates
        ntv3_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if ('nucleotide_embedding' in name or 'token_encoder' in name) and 'model' in name:
                    # NTv3 model parameters - use smaller learning rate
                    ntv3_params.append(param)
                else:
                    # Other parameters (Mamba, classification head, etc.)
                    other_params.append(param)
        
        if not ntv3_params and not other_params:
            raise ValueError("No trainable parameters! Please check model configuration.")
        
        # Set different learning rates for different parameter groups
        optimizer_grouped_parameters = []
        
        # NTv3 parameters: smaller learning rate
        if ntv3_params:
            optimizer_grouped_parameters.append({
                "params": ntv3_params,
                "lr": self.config.learning_rate * self.config.ntv3_learning_rate_multiplier,  # 10% of base learning rate
                "weight_decay": self.config.weight_decay,
            })
        
        # Other parameters: normal learning rate
        if other_params:
            optimizer_grouped_parameters.append({
                "params": other_params,
                "lr": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
            })
        
        print(f"  Optimizer parameter groups:")
        print(f"    NTv3 parameters: {len(ntv3_params)} groups, learning rate: {self.config.learning_rate * config.ntv3_learning_rate_multiplier}")
        print(f"    Other parameters: {len(other_params)} groups, learning rate: {self.config.learning_rate}")
        
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Learning rate scheduler
        # steps_per_epoch <= 0 means "auto": resolved from the actual number
        # of train batches per epoch once loaders exist (set in train()).
        scheduler_steps = self.config.steps_per_epoch
        if scheduler_steps <= 0:
            scheduler_steps = getattr(self, "_steps_per_epoch", None) or 1
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.onecycle_max_lr,
            epochs=self.config.num_epochs,
            steps_per_epoch=scheduler_steps,
            pct_start=self.config.onecycle_pct_start,
            anneal_strategy='cos'
        )
        print(f"  Scheduler steps_per_epoch: {scheduler_steps}")
        
        return optimizer, scheduler
    
    # The following methods remain unchanged...
    def calculate_metrics(self, labels, predictions, probabilities):
        """Calculate evaluation metrics"""
        if len(labels) == 0:
            return {
                'acc': 0.0,
                'f1': 0.0,
                'auc': 0.0,
                'mcc': 0.0,
                'samples': 0
            }
        
        try:
            labels_np = np.array(labels).flatten()
            predictions_np = np.array(predictions).flatten()
            probabilities_np = np.array(probabilities).flatten()
            
            if len(labels_np) != len(predictions_np) or len(labels_np) != len(probabilities_np):
                min_len = min(len(labels_np), len(predictions_np), len(probabilities_np))
                labels_np = labels_np[:min_len]
                predictions_np = predictions_np[:min_len]
                probabilities_np = probabilities_np[:min_len]
            
            acc = accuracy_score(labels_np, predictions_np)
            f1 = f1_score(labels_np, predictions_np, average="binary", zero_division=0)
            
            unique_labels = np.unique(labels_np)
            if len(unique_labels) > 1:
                try:
                    auc = roc_auc_score(labels_np, probabilities_np)
                except Exception:
                    auc = 0.0
            else:
                auc = 0.0
                
            if len(unique_labels) > 1:
                try:
                    mcc = matthews_corrcoef(labels_np, predictions_np)
                except Exception:
                    mcc = 0.0
            else:
                mcc = 0.0
            
            return {
                'acc': float(acc),
                'f1': float(f1),
                'auc': float(auc),
                'mcc': float(mcc),
                'samples': len(labels_np)
            }
            
        except Exception as e:
            print(f"❌ Failed to calculate metrics: {str(e)}")
            return {
                'acc': 0.0,
                'f1': 0.0,
                'auc': 0.0,
                'mcc': 0.0,
                'samples': 0
            }

    def train_epoch(self, model, train_loader, class_criterion, contrastive_criterion, 
                    optimizer, scheduler, scaler, epoch):
        """Train one epoch"""
        model.train()
        
        epoch_loss = 0.0
        epoch_class_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_path_loss = 0.0
        
        # Metric collection
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        original_labels = []
        original_predictions = []
        original_probabilities = []
        
        semantic_labels = []
        semantic_predictions = []
        semantic_probabilities = []
        
        confusion_labels = []
        confusion_predictions = []
        confusion_probabilities = []
        
        # Clean GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Use tqdm to show progress
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move to device
                batch = self._move_batch_to_device(batch)
                
                batch_labels = batch['labels']
                batch_size = batch_labels.size(0)
                positive_mask = (batch_labels == 1)
                positive_indices = torch.where(positive_mask)[0]
                
                # Prepare model inputs
                original_sequences = batch['sequences']
                semantic_sequences = batch.get('semantic_sequences', None)
                confusion_sequences = batch.get('confusion_sequences', None)
                random_mutation_sequences = batch.get('random_mutation_sequences', None)
                
                # Mixed precision training
                with autocast(device_type='cuda', enabled=True):
                    # Forward pass
                    outputs = model(
                        original_sequences=original_sequences,
                        labels=batch['labels'],
                        semantic_sequences=semantic_sequences,
                        confusion_sequences=confusion_sequences,
                        random_mutation_sequences=random_mutation_sequences,
                        training_mode=True
                    )
                    
                    # Unpack outputs
                    orig_class_pred, semantic_class_pred, confusion_class_pred, total_aux_loss, orig_contrastive_feat, all_contrastive_feat = outputs
                    
                    # ===== Calculate classification loss =====
                    if orig_class_pred.dim() == 2 and orig_class_pred.size(1) == 1:
                        orig_class_pred_flat = orig_class_pred.squeeze(1)
                    else:
                        orig_class_pred_flat = orig_class_pred

                    orig_class_loss = class_criterion(orig_class_pred_flat, batch['labels'])
                    # Pure-task protocol uses coefficient 1.0 (baseline BCE scale);
                    # the variant BCE terms below stay inert without variant views.
                    total_class_loss = orig_class_loss * (1.0 if self.pure_mode else self.loss_weights["orig"])

                    # Variant classification loss: applies to every record that
                    # carries the flavor. With symmetric augmentation negatives
                    # carry label-0 variants; each record's own label is the
                    # target (legacy data: subset of positives, target 1 —
                    # identical to the previous positive-only behavior).
                    sem_variant_list = variant_indices(semantic_sequences, batch_size)
                    conf_variant_list = variant_indices(confusion_sequences, batch_size)
                    sem_variant_idx = (torch.tensor(sem_variant_list, device=self.device, dtype=torch.long)
                                       if sem_variant_list else None)
                    conf_variant_idx = (torch.tensor(conf_variant_list, device=self.device, dtype=torch.long)
                                        if conf_variant_list else None)

                    if semantic_class_pred is not None and sem_variant_idx is not None:
                        semantic_class_pred_v = semantic_class_pred[sem_variant_idx]
                        if semantic_class_pred_v.dim() == 2 and semantic_class_pred_v.size(1) == 1:
                            semantic_class_pred_v = semantic_class_pred_v.squeeze(1)
                        if semantic_class_pred_v.size(0) > 0:
                            semantic_class_loss = class_criterion(
                                semantic_class_pred_v,
                                batch['labels'][sem_variant_idx]
                            )
                            total_class_loss += semantic_class_loss * self.loss_weights["sem"]

                    if confusion_class_pred is not None and conf_variant_idx is not None:
                        confusion_class_pred_v = confusion_class_pred[conf_variant_idx]
                        if confusion_class_pred_v.dim() == 2 and confusion_class_pred_v.size(1) == 1:
                            confusion_class_pred_v = confusion_class_pred_v.squeeze(1)
                        if confusion_class_pred_v.size(0) > 0:
                            confusion_class_loss = class_criterion(
                                confusion_class_pred_v,
                                batch['labels'][conf_variant_idx]
                            )
                            total_class_loss += confusion_class_loss * self.loss_weights["conf"]
                    
                    # ===== Calculate contrastive loss =====
                    contrastive_loss = torch.tensor(0.0, device=self.device)
                    
                    if (orig_contrastive_feat is not None and 
                        all_contrastive_feat is not None and 
                        len(positive_indices) > 0):
                        
                        has_semantic = semantic_sequences is not None and any(len(s) > 0 for s in semantic_sequences)
                        has_confusion = confusion_sequences is not None and any(len(s) > 0 for s in confusion_sequences)
                        has_random_mutation = random_mutation_sequences is not None and any(len(s) > 0 for s in random_mutation_sequences)
                        
                        if has_semantic and has_confusion and has_random_mutation:
                            orig_contrastive_feat_positive = orig_contrastive_feat[positive_indices]
                            
                            if all_contrastive_feat is not None:
                                split_size = batch_size
                                split_features = torch.split(all_contrastive_feat, split_size, dim=0)
                                
                                if len(split_features) >= 4:
                                    semantic_contrastive_feat_all = split_features[1]
                                    confusion_contrastive_feat_all = split_features[2]
                                    random_mutation_contrastive_feat_all = split_features[3]
                                    
                                    semantic_contrastive_feat_positive = semantic_contrastive_feat_all[positive_indices]
                                    confusion_contrastive_feat_positive = confusion_contrastive_feat_all[positive_indices]
                                    random_mutation_contrastive_feat_positive = random_mutation_contrastive_feat_all[positive_indices]
                                    
                                    if (orig_contrastive_feat_positive.size(0) > 0 and 
                                        semantic_contrastive_feat_positive.size(0) > 0 and
                                        confusion_contrastive_feat_positive.size(0) > 0 and
                                        random_mutation_contrastive_feat_positive.size(0) > 0):
                                        
                                        contrastive_loss = contrastive_criterion(
                                            anchor_feat=orig_contrastive_feat_positive.clone(),
                                            positive_feats=[
                                                semantic_contrastive_feat_positive.clone(),
                                                confusion_contrastive_feat_positive.clone()
                                            ],
                                            negative_feat=random_mutation_contrastive_feat_positive.clone()
                                        )
                    
                    # ===== Total loss =====
                    if self.pure_mode:
                        # Plain original-sample BCE: no contrastive, no path
                        # KL/entropy supervision, no 1.4 renormalization.
                        total_loss = orig_class_loss
                    else:
                        total_loss = (total_class_loss
                                      + contrastive_loss * self.loss_weights["contrast"]
                                      + total_aux_loss * self.loss_weights["path"]) / self.loss_weights["norm"]
                
                # Backward pass
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(total_loss).backward()
                
                # Gradient clipping
                if self.config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Accumulate losses
                batch_size_actual = batch['labels'].size(0)
                epoch_loss += total_loss.item() * batch_size_actual
                epoch_class_loss += total_class_loss.item() * batch_size_actual
                epoch_contrastive_loss += contrastive_loss.item() * batch_size_actual
                
                if isinstance(total_aux_loss, torch.Tensor):
                    epoch_path_loss += total_aux_loss.item() * batch_size_actual
                else:
                    epoch_path_loss += total_aux_loss * batch_size_actual
                
                # Metric collection
                batch_labels_cpu = batch['labels'].cpu().detach().numpy()
                pos_indices_np = positive_indices.cpu().numpy() if len(positive_indices) > 0 else np.array([])
                
                # 1. Original sequence predictions
                orig_probs = torch.sigmoid(orig_class_pred).cpu().detach().numpy()
                if orig_probs.ndim == 2 and orig_probs.shape[1] == 1:
                    orig_probs = orig_probs.flatten()
                
                orig_preds = (orig_probs > 0.5).astype(int)
                
                # Collect original sequence metrics
                original_labels.extend(batch_labels_cpu.tolist())
                original_predictions.extend(orig_preds.tolist())
                original_probabilities.extend(orig_probs.tolist())
                
                # 2. Overall metrics
                all_labels.extend(batch_labels_cpu.tolist())
                all_predictions.extend(orig_preds.tolist())
                all_probabilities.extend(orig_probs.tolist())
                
                # 3. Variant metrics (every record carrying the flavor; with
                # symmetric augmentation this spans both classes)
                if sem_variant_list:
                    if semantic_class_pred is not None:
                        semantic_probs = torch.sigmoid(semantic_class_pred[sem_variant_idx]).cpu().detach().numpy()
                        if semantic_probs.ndim == 2 and semantic_probs.shape[1] == 1:
                            semantic_probs = semantic_probs.flatten()

                        semantic_preds = (semantic_probs > 0.5).astype(int)
                        semantic_labels_batch = batch_labels_cpu[sem_variant_list]

                        all_labels.extend(semantic_labels_batch.tolist())
                        all_predictions.extend(semantic_preds.tolist())
                        all_probabilities.extend(semantic_probs.tolist())

                        semantic_labels.extend(semantic_labels_batch.tolist())
                        semantic_predictions.extend(semantic_preds.tolist())
                        semantic_probabilities.extend(semantic_probs.tolist())

                if conf_variant_list:
                    if confusion_class_pred is not None:
                        confusion_probs = torch.sigmoid(confusion_class_pred[conf_variant_idx]).cpu().detach().numpy()
                        if confusion_probs.ndim == 2 and confusion_probs.shape[1] == 1:
                            confusion_probs = confusion_probs.flatten()

                        confusion_preds = (confusion_probs > 0.5).astype(int)
                        confusion_labels_batch = batch_labels_cpu[conf_variant_list]

                        all_labels.extend(confusion_labels_batch.tolist())
                        all_predictions.extend(confusion_preds.tolist())
                        all_probabilities.extend(confusion_probs.tolist())

                        confusion_labels.extend(confusion_labels_batch.tolist())
                        confusion_predictions.extend(confusion_preds.tolist())
                        confusion_probabilities.extend(confusion_probs.tolist())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss.item(),
                    'class': total_class_loss.item(),
                    'contrast': contrastive_loss.item(),
                    'aux': total_aux_loss.item() if isinstance(total_aux_loss, torch.Tensor) else total_aux_loss,
                    'pos': len(positive_indices)
                })
                
                # Periodically clean cache
                if batch_idx % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate epoch metrics
            epoch_samples = len(train_loader.dataset)
            avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
            avg_class_loss = epoch_class_loss / epoch_samples if epoch_samples > 0 else 0
            avg_contrastive_loss = epoch_contrastive_loss / epoch_samples if epoch_samples > 0 else 0
            avg_path_loss = epoch_path_loss / epoch_samples if epoch_samples > 0 else 0
            
            # Calculate various metrics
            overall_metrics = self.calculate_metrics(all_labels, all_predictions, all_probabilities)
            original_metrics = self.calculate_metrics(original_labels, original_predictions, original_probabilities)
            semantic_metrics = self.calculate_metrics(semantic_labels, semantic_predictions, semantic_probabilities)
            confusion_metrics = self.calculate_metrics(confusion_labels, confusion_predictions, confusion_probabilities)
            
            # Calculate combined metric (pure-task: original ACC only — the
            # variant terms are undefined without views)
            if self.pure_mode:
                combined_acc = original_metrics['acc']
            else:
                variant_weight = getattr(self.config, 'variant_acc_weight', 0.7)
                combined_acc = original_metrics['acc'] * (1 - variant_weight) + \
                            (semantic_metrics['acc'] + confusion_metrics['acc']) / 2 * variant_weight
            
            return {
                'loss': avg_loss,
                'class_loss': avg_class_loss,
                'contrastive_loss': avg_contrastive_loss,
                'path_loss': avg_path_loss,
                'overall': overall_metrics,
                'original': original_metrics,
                'semantic': semantic_metrics,
                'confusion': confusion_metrics,
                'combined_acc': combined_acc,
                'samples': epoch_samples
            }

    def validate(self, model, val_loader, class_criterion, contrastive_criterion):
        """Validate model"""
        model.eval()
        
        val_loss = 0.0
        val_class_loss = 0.0
        val_contrastive_loss = 0.0
        val_path_loss = 0.0
        
        # Separate storage for different metric sets
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        original_labels = []
        original_predictions = []
        original_probabilities = []
        
        semantic_labels = []
        semantic_predictions = []
        semantic_probabilities = []
        
        confusion_labels = []
        confusion_predictions = []
        confusion_probabilities = []
        
        # Clean GPU cache before validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Fix label format
                if 'labels' in batch:
                    labels = batch['labels']
                    if isinstance(labels, torch.Tensor):
                        if labels.dim() == 0:
                            batch['labels'] = labels.unsqueeze(0)
                    else:
                        batch['labels'] = torch.tensor([float(labels)], dtype=torch.float32)
                
                batch = self._move_batch_to_device(batch)
                
                batch_labels = batch['labels']
                if batch_labels.dim() == 0:
                    batch_labels = batch_labels.unsqueeze(0)
                    batch['labels'] = batch_labels
                
                batch_size = batch_labels.size(0)
                positive_mask = (batch_labels == 1)
                positive_indices = torch.where(positive_mask)[0]
                
                # Prepare model inputs
                original_sequences = batch['sequences']
                semantic_sequences = batch.get('semantic_sequences', None)
                confusion_sequences = batch.get('confusion_sequences', None)
                random_mutation_sequences = batch.get('random_mutation_sequences', None)
                
                # Forward pass
                outputs = model(
                    original_sequences=original_sequences,
                    labels=batch['labels'],
                    semantic_sequences=semantic_sequences,
                    confusion_sequences=confusion_sequences,
                    random_mutation_sequences=random_mutation_sequences,
                    training_mode=False
                )
                
                orig_class_pred, semantic_class_pred, confusion_class_pred, total_aux_loss, orig_contrastive_feat, all_contrastive_feat = outputs
                
                # ===== Calculate loss =====
                if orig_class_pred.dim() == 2 and orig_class_pred.size(1) == 1:
                    orig_class_pred_flat = orig_class_pred.squeeze(1)
                else:
                    orig_class_pred_flat = orig_class_pred
                
                orig_class_loss = class_criterion(orig_class_pred_flat, batch['labels'])
                total_class_loss = orig_class_loss * (1.0 if self.pure_mode else self.loss_weights["orig"])

                # Variant losses follow every record carrying the flavor
                # (symmetric augmentation: negatives carry label-0 variants)
                sem_variant_list = variant_indices(semantic_sequences, batch_size)
                conf_variant_list = variant_indices(confusion_sequences, batch_size)
                sem_variant_idx = (torch.tensor(sem_variant_list, device=self.device, dtype=torch.long)
                                   if sem_variant_list else None)
                conf_variant_idx = (torch.tensor(conf_variant_list, device=self.device, dtype=torch.long)
                                    if conf_variant_list else None)

                if semantic_class_pred is not None and sem_variant_idx is not None:
                    semantic_class_pred_v = semantic_class_pred[sem_variant_idx]
                    if semantic_class_pred_v.dim() == 2 and semantic_class_pred_v.size(1) == 1:
                        semantic_class_pred_v = semantic_class_pred_v.squeeze(1)
                    if semantic_class_pred_v.size(0) > 0:
                        semantic_class_loss = class_criterion(
                            semantic_class_pred_v,
                            batch['labels'][sem_variant_idx]
                        )
                        total_class_loss += semantic_class_loss * 0.1

                if confusion_class_pred is not None and conf_variant_idx is not None:
                    confusion_class_pred_v = confusion_class_pred[conf_variant_idx]
                    if confusion_class_pred_v.dim() == 2 and confusion_class_pred_v.size(1) == 1:
                        confusion_class_pred_v = confusion_class_pred_v.squeeze(1)
                    if confusion_class_pred_v.size(0) > 0:
                        confusion_class_loss = class_criterion(
                            confusion_class_pred_v,
                            batch['labels'][conf_variant_idx]
                        )
                        total_class_loss += confusion_class_loss * 0.1
                
                # Contrastive loss
                contrastive_loss = torch.tensor(0.0, device=self.device)
                
                if len(positive_indices) > 0:
                    has_semantic = semantic_sequences is not None and any(len(s) > 0 for s in semantic_sequences)
                    has_confusion = confusion_sequences is not None and any(len(s) > 0 for s in confusion_sequences)
                    has_random_mutation = random_mutation_sequences is not None and any(len(s) > 0 for s in random_mutation_sequences)
                    
                    if has_semantic and has_confusion and has_random_mutation:
                        orig_contrastive_feat_positive = orig_contrastive_feat[positive_indices]
                        
                        batch_size_val = batch['labels'].size(0)
                        split_size = batch_size_val
                        split_features = torch.split(all_contrastive_feat, split_size, dim=0)
                        
                        if len(split_features) >= 4:
                            semantic_contrastive_feat_all = split_features[1]
                            confusion_contrastive_feat_all = split_features[2]
                            random_mutation_contrastive_feat_all = split_features[3]
                            
                            semantic_contrastive_feat_positive = semantic_contrastive_feat_all[positive_indices]
                            confusion_contrastive_feat_positive = confusion_contrastive_feat_all[positive_indices]
                            random_mutation_contrastive_feat_positive = random_mutation_contrastive_feat_all[positive_indices]
                            
                            if orig_contrastive_feat_positive.size(0) > 0:
                                contrastive_loss = contrastive_criterion(
                                    anchor_feat=orig_contrastive_feat_positive,
                                    positive_feats=[semantic_contrastive_feat_positive, confusion_contrastive_feat_positive],
                                    negative_feat=random_mutation_contrastive_feat_positive
                                )
                
                if self.pure_mode:
                    total_loss = orig_class_loss
                else:
                    total_loss = (total_class_loss
                                  + contrastive_loss * self.loss_weights["contrast"]
                                  + total_aux_loss * self.loss_weights["path"]) / self.loss_weights["norm"]
                
                # Accumulate losses
                batch_size_val = batch['labels'].size(0)
                val_loss += total_loss.item() * batch_size_val
                val_class_loss += total_class_loss.item() * batch_size_val
                val_contrastive_loss += contrastive_loss.item() * batch_size_val
                val_path_loss += total_aux_loss.item() * batch_size_val
                
                # ===== Metric collection =====
                batch_labels_cpu = batch['labels'].cpu().detach().numpy()
                pos_indices_np = positive_indices.cpu().numpy() if len(positive_indices) > 0 else np.array([])
                
                # 1. Original sequence metrics
                orig_probs = torch.sigmoid(orig_class_pred).cpu().detach().numpy()
                if orig_probs.ndim == 2 and orig_probs.shape[1] == 1:
                    orig_probs = orig_probs.flatten()
                
                orig_preds = (orig_probs > 0.5).astype(int)
                
                original_labels.extend(batch_labels_cpu.tolist())
                original_predictions.extend(orig_preds.tolist())
                original_probabilities.extend(orig_probs.tolist())
                
                # 2. Overall metrics
                all_labels.extend(batch_labels_cpu.tolist())
                all_predictions.extend(orig_preds.tolist())
                all_probabilities.extend(orig_probs.tolist())
                
                # 3. Variant metrics (every record carrying the flavor)
                if sem_variant_list:
                    if semantic_class_pred is not None:
                        semantic_probs = torch.sigmoid(semantic_class_pred[sem_variant_idx]).cpu().detach().numpy()
                        if semantic_probs.ndim == 2 and semantic_probs.shape[1] == 1:
                            semantic_probs = semantic_probs.flatten()

                        semantic_preds = (semantic_probs > 0.5).astype(int)
                        semantic_labels_batch = batch_labels_cpu[sem_variant_list]

                        all_labels.extend(semantic_labels_batch.tolist())
                        all_predictions.extend(semantic_preds.tolist())
                        all_probabilities.extend(semantic_probs.tolist())

                        semantic_labels.extend(semantic_labels_batch.tolist())
                        semantic_predictions.extend(semantic_preds.tolist())
                        semantic_probabilities.extend(semantic_probs.tolist())

                if conf_variant_list:
                    if confusion_class_pred is not None:
                        confusion_probs = torch.sigmoid(confusion_class_pred[conf_variant_idx]).cpu().detach().numpy()
                        if confusion_probs.ndim == 2 and confusion_probs.shape[1] == 1:
                            confusion_probs = confusion_probs.flatten()

                        confusion_preds = (confusion_probs > 0.5).astype(int)
                        confusion_labels_batch = batch_labels_cpu[conf_variant_list]

                        all_labels.extend(confusion_labels_batch.tolist())
                        all_predictions.extend(confusion_preds.tolist())
                        all_probabilities.extend(confusion_probs.tolist())

                        confusion_labels.extend(confusion_labels_batch.tolist())
                        confusion_predictions.extend(confusion_preds.tolist())
                        confusion_probabilities.extend(confusion_probs.tolist())
        
        # ===== Calculate validation metrics =====
        val_samples = sum(1 for _ in val_loader) * self.config.batch_size if len(val_loader) > 0 else 0
        avg_loss = val_loss / val_samples if val_samples > 0 else 0
        avg_class_loss = val_class_loss / val_samples if val_samples > 0 else 0
        avg_contrastive_loss = val_contrastive_loss / val_samples if val_samples > 0 else 0
        avg_path_loss = val_path_loss / val_samples if val_samples > 0 else 0
        
        # Calculate various metrics
        overall_metrics = self.calculate_metrics(all_labels, all_predictions, all_probabilities)
        original_metrics = self.calculate_metrics(original_labels, original_predictions, original_probabilities)
        semantic_metrics = self.calculate_metrics(semantic_labels, semantic_predictions, semantic_probabilities)
        confusion_metrics = self.calculate_metrics(confusion_labels, confusion_predictions, confusion_probabilities)
        
        # Print debug information
        print(f"\n🔍 Validation set statistics:")
        print(f"  Original sequence samples: {len(original_labels)}")
        print(f"  Semantic variant samples: {len(semantic_labels)}")
        print(f"  Confusion variant samples: {len(confusion_labels)}")
        print(f"  Total samples: {len(all_labels)}")
        
        # Calculate combined ACC (pure-task: original ACC only)
        if self.pure_mode:
            combined_acc = original_metrics['acc']
        else:
            variant_weight = getattr(self.config, 'variant_acc_weight', 0.35)
            combined_acc = original_metrics['acc'] * (1 - variant_weight) + \
                        (semantic_metrics['acc'] + confusion_metrics['acc']) / 2 * variant_weight
        
        return {
            'loss': avg_loss,
            'class_loss': avg_class_loss,
            'contrastive_loss': avg_contrastive_loss,
            'path_loss': avg_path_loss,
            'overall': overall_metrics,
            'original': original_metrics,
            'semantic': semantic_metrics,
            'confusion': confusion_metrics,
            'combined_acc': combined_acc,
            'samples': val_samples
        }
    
    def _move_batch_to_device(self, batch):
        """Move batch data to device"""
        batch_gpu = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_gpu[key] = value.to(self.device, non_blocking=True)
            else:
                batch_gpu[key] = value
        return batch_gpu
    
    def save_model(self, model, optimizer, scheduler, epoch, val_metrics, path):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_metrics': val_metrics,
            'config': self.config.__dict__,
            # Config defaults are class attributes and are absent from __dict__.
            'random_seeds': self.get_seed_config(),
            'data_split': self.data_split_summary,
            'model_config': {
                'transformer_model_repo': model.transformer_model_repo,
                'embedding_dim': model.embedding_dim,
                'd_model': model.d_model,
                'n_layer': model.n_layer,
                'block_type': model.block_type,
                'classifier_head': model.classifier_head,
            'readout_pool': getattr(self.config, 'readout_pool', 'attention'),
            'readout_fusion': getattr(self.config, 'readout_fusion', 'linear'),
            'readout_rank': getattr(self.config, 'readout_rank', 64),
            'readout_source': getattr(self.config, 'readout_source', 'input'),
                'zero_internal_dropout': getattr(model, 'zero_internal_dropout', False),
                'use_path_selection': model.use_path_selection,
                'pure_task_mode': self.pure_mode,
                # Rebuild-critical: the embedding truncates inputs to
                # max_seq_len at tokenization, so eval must reuse the
                # training value (2026-09-20 eval-pipeline fix).
                'max_seq_len': getattr(self.config, 'max_seq_len', 2048),
                'dropout_rate': getattr(self.config, 'dropout_rate', 0.2),
            },
            'cache_stats': model.get_cache_stats()
        }
        
        torch.save(checkpoint, path)
        print(f"💾 Model saved to: {path}")
    
    def get_seed_config(self):
        """Return explicit seed metadata, including defaults on the config class."""
        return {
            'seed': self.config.seed,
            'sampling_seed': getattr(self.config, 'sampling_seed', 42),
            'split_seed': DATA_SPLIT_SEED,
        }

    def train(self, variants_data):
        """Main training function"""
        print(f"\n🎯 Starting fully trainable Nucleotide Transformer v3 training pipeline")

        # Seed here, not on config import: this also covers direct trainer.train()
        # calls and applies overrides before any loader or model is constructed.
        seed_training(self.config.seed)
        print(f"  Random seeds: {self.get_seed_config()}")
        
        # 1. Prepare data
        train_loader, val_loader = self.prepare_data(variants_data)

        # Resolve auto steps_per_epoch (config value <= 0) from the real
        # number of train batches so OneCycleLR completes its schedule.
        if getattr(self.config, "steps_per_epoch", 0) <= 0:
            self._steps_per_epoch = max(1, len(train_loader))
        
        # 2. Create model
        model = self.create_model()
        
        # 3. Create loss functions
        class_criterion, contrastive_criterion = self.create_loss_functions()
        
        # 4. Create optimizer
        optimizer, scheduler = self.create_optimizer(model)
        
        # 5. Mixed precision training
        scaler = GradScaler(enabled=True)
        
        # 6. Training loop
        best_combined_acc = 0.0
        early_stop_counter = 0
        early_stopped = False
        
        train_metrics_history = []
        val_metrics_history = []
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Train
            train_metrics = self.train_epoch(
                model, train_loader, class_criterion, contrastive_criterion,
                optimizer, scheduler, scaler, epoch
            )
            
            # Validate
            val_metrics = self.validate(model, val_loader, class_criterion, contrastive_criterion)
            
            train_metrics_history.append(train_metrics)
            val_metrics_history.append(val_metrics)
            
            epoch_time = time.time() - epoch_start_time
            current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else 0
            
            print(f"\n📊 Epoch {epoch+1}/{self.config.num_epochs} Summary:")
            print(f"  Time: {epoch_time:.1f}s | Learning rate: {current_lr:.2e}")
            print(f"  Training loss: {train_metrics['loss']:.4f} | Validation loss: {val_metrics['loss']:.4f}")
            print(f"  Training combined ACC: {train_metrics['combined_acc']:.4f} | Validation combined ACC: {val_metrics['combined_acc']:.4f}")
            
            print(f"\n  📈 Overall metrics:")
            print(f"    Training ACC: {train_metrics['overall']['acc']:.4f} | Validation ACC: {val_metrics['overall']['acc']:.4f}")
            print(f"    Training F1: {train_metrics['overall']['f1']:.4f} | Validation F1: {val_metrics['overall']['f1']:.4f}")
            print(f"    Training AUC: {train_metrics['overall']['auc']:.4f} | Validation AUC: {val_metrics['overall']['auc']:.4f}")
            
            print(f"\n  🧬 Original sequence metrics:")
            print(f"    Training ACC: {train_metrics['original']['acc']:.4f} | Validation ACC: {val_metrics['original']['acc']:.4f}")
            
            print(f"\n  🔍 Specificity metrics:")
            print(f"    Semantic variant ACC: {train_metrics['semantic']['acc']:.4f} | {val_metrics['semantic']['acc']:.4f}")
            print(f"    Confusion variant ACC: {train_metrics['confusion']['acc']:.4f} | {val_metrics['confusion']['acc']:.4f}")
            
            # Print cache statistics
            cache_stats = model.get_cache_stats()
            print(f"\n  💾 Cache statistics: Hit rate: {cache_stats['hit_rate']:.2%} | Cache size: {cache_stats['cache_size']:,}")
            
            # Save best model
            if val_metrics['combined_acc'] > best_combined_acc:
                best_combined_acc = val_metrics['combined_acc']
                early_stop_counter = 0
                
                best_model_path = os.path.join(self.config.output_dir, "best_nucleotide_v3_mamba_model.pth")
                self.save_model(model, optimizer, scheduler, epoch + 1, val_metrics, best_model_path)
                
                print(f"\n✅ New best combined ACC: {best_combined_acc:.4f}")
            else:
                early_stop_counter += 1
                print(f"\n⚠️  Combined ACC not improved, early stop counter: {early_stop_counter}/{self.config.early_stop_patience}")
            
            if early_stop_counter >= self.config.early_stop_patience:
                print(f"\n🛑 Early stopping triggered, stopping training")
                early_stopped = True
                break
        
        # Save final model
        final_model_path = os.path.join(self.config.output_dir, "final_nucleotide_v3_mamba_model.pth")
        self.save_model(model, optimizer, scheduler, self.config.num_epochs, 
                       val_metrics_history[-1] if val_metrics_history else {}, final_model_path)
        
        # Save training log
        log_data = {
            'config': self.config.__dict__,
            'random_seeds': self.get_seed_config(),
            'data_split': self.data_split_summary,
            'train_metrics': train_metrics_history,
            'val_metrics': val_metrics_history,
            'best_combined_acc': best_combined_acc,
            'early_stopped': early_stopped,
            'total_epochs': len(train_metrics_history),
            'realtime_processing': True,
            'cache_stats_final': model.get_cache_stats(),
            'transformer_config': {
                'model_repo': model.transformer_model_repo,
                'freeze_transformer': model.freeze_transformer,
                'trainable': 'fully_trainable'
            }
        }
        
        log_path = os.path.join(self.config.output_dir, "nucleotide_v3_training_log.json")
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        print(f"\n📄 Training log saved to: {log_path}")
        
        # 7. Call visualization module (optional)
        try:
            from training_visualizer import integrate_visualization
            print(f"\n🎨 Generating training visualization charts...")
            integrate_visualization(self, train_metrics_history, val_metrics_history, self.config, early_stopped)
            print(f"✅ Visualization charts generated successfully")
        except ImportError:
            # Optional plotting module is not part of the release.
            pass
        except Exception as e:
            print(f"⚠️  Visualization generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Clear cache
        model.clear_cache()
        
        return model, train_metrics_history, val_metrics_history


def train_nucleotide(variants_data):
    """Nucleotide Transformer v3 training main function"""
    trainer = NucleotideMambaTrainer(config)
    
    try:
        model, train_history, val_history = trainer.train(variants_data)
        print(f"\n🎉 Nucleotide Transformer v3 training completed!")
        
        if val_history:
            final_val = val_history[-1]
            print(f"\n📈 Final validation set metrics:")
            print(f"  Combined ACC: {final_val['combined_acc']:.4f}")
            print(f"  Overall ACC: {final_val['overall']['acc']:.4f}")
            print(f"  Overall F1: {final_val['overall']['f1']:.4f}")
            print(f"  Overall AUC: {final_val['overall']['auc']:.4f}")
            print(f"  Overall MCC: {final_val['overall']['mcc']:.4f}")
            print(f"  Original ACC: {final_val['original']['acc']:.4f}")
            print(f"  Semantic ACC: {final_val['semantic']['acc']:.4f}")
            print(f"  Confusion ACC: {final_val['confusion']['acc']:.4f}")
        
        return model
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
