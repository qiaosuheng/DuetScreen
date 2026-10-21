import torch
import os
from typing import List

class Config:
    # ========== Basic Configuration ==========
    model_type = "nucleotide_v3_mamba"  # Current model type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Direct device specification
    
    # ========== Run Mode Configuration ==========
    run_mode = "queue"  # "queue" (recommended): cache-aware multi-seed training
                        # + evaluation queue. Legacy single-run modes remain:
                        # "preprocess", "generate", "train", "all", "train-only"
    
    # ========== File Configuration ==========
    use_existing_variants = False  # Whether to use existing variant file
    variants_file_path = "results_nucleotide_v3_mamba/variants_data.pkl"  # Variant file path
    output_dir = "results_nmi_mixed_seed42"  # Output directory (one per seed/pool)
    
    # ========== Data Configuration ==========
    max_seq_length = 2048  # Maximum DNA sequence length ⚠️ Changed to 2048
    min_segment_len = 30  # Minimum fragment length
    positive_fasta = "train_data/VFDB_setB_nt.fasta"  # Positive samples file
    negative_fasta = "train_data/non_pathogenic_regions.fasta"  # Negative samples file
    # 0 = preserve the delivered class composition (decision for the NMI
    # revision; the release is pre-balanced at both record and segment level).
    # Set 3.0 to restore the legacy negative-downsampling behavior.
    downsample_ratio = 0.0

    # ========== Prepared-Split Mode (nmi_dataset_release) ==========
    # True: read the delivered train/val FASTA pair directly (one file per
    # split, labels encoded in the record ID prefix pos_/neg_) and skip the
    # internal 8:1:1 re-split. Validation runs on the delivered val records;
    # positives in both splits receive generated variants (variant content is
    # derived from its own record only). external isolation is enforced by
    # data_split.verify_prepared_split against independent_test_files.
    use_prepared_split = True
    prepared_train_fasta = "train_data_nmi/mixed/train.fasta"
    prepared_val_fasta = "train_data_nmi/mixed/val.fasta"
    # Exact-content dedup of segments. The release explicitly keeps
    # record-level redundancy inside each split (content groups never cross
    # splits; see docs), so prepared mode preserves the delivered composition.
    remove_duplicate_segments = False

    # ========== Training Queue (run_mode="queue") ==========
    # One shared variant cache per pool; membership is fixed by the delivered
    # files, so the cache is seed-independent and generated at most once.
    training_seeds = [42, 43, 44, 45, 46]
    # Symmetric augmentation: negatives also receive semantic/confusion
    # variants (label 0) to break the "synthetic flavor => positive" shortcut
    #   mixed     = legacy positive-only variants (before arm)
    #   mixed_sym = symmetric augmentation (after arm)
    symmetric_variants = True
    # Pure-task protocol (arm C, control specification) — completed 2026-09-20:
    # test 0.846±0.017 (≈B+), proving the dual-path stack is innocent and the
    # auxiliary objective package caused O's −0.09. Disabled for the
    # reweighted arm below.
    pure_task_mode = False
    # Reweighted objective (arm "rw", internal control specification): the
    # original-sample task keeps 1 - aux_budget of the loss; the auxiliary
    # terms (view BCE / contrastive / path supervision) share aux_budget
    # proportionally to their legacy weights, no 1.4 renormalization.
    # Framing: variants are demoted to "biologically-motivated perturbation
    # samples"; augmentation must not dominate the primary task loss.
    objective_reweight = True
    aux_budget = 0.1
    # pool_name convention (architecture-overhaul, 2026-09-18; arm C added
    # 2026-09-19 after the matched-control plan):
    #   <dataset-pool>_<protocol>_<arch-tag>
    #   mixed_sym      = mixed pool + symmetric augmentation + Mamba core (previous arm)
    #   mixed_sym_tr   = mixed pool + symmetric augmentation + Transformer core
    #   mixed_sym_conv = mixed pool + symmetric augmentation + dilated-Conv core
    #   mixed_pure     = mixed pool + pure-task protocol + Mamba core (arm C)
    #   mixed_sym_rw   = mixed pool + symmetric augmentation + reweighted 0.9/0.1 objective (this arm)
    pool_name = "mixed_sym_rw"   # naming component for runs/ and evaluations/
    results_root = "runs"          # <results_root>/<pool_name>/cache + seed<N>/
    evaluations_root = "evaluations"  # <evaluations_root>/<pool_name>/seed<N>/<set>/
    eval_threshold = 0.5
    eval_batch_size = 64
    eval_sets = {
        "test": os.path.join(os.path.dirname(__file__), "train_data_nmi", "mixed", "eval_test.fasta"),
        "species": os.path.join(os.path.dirname(__file__), "train_data_nmi", "mixed", "eval_generalization_species.fasta"),
        "family": os.path.join(os.path.dirname(__file__), "train_data_nmi", "mixed", "eval_generalization_family.fasta"),
    }
    # Change only seed between independent training runs (e.g. 42, 43, 44, 45, 46).
    # It controls Python/NumPy and PyTorch CPU/CUDA training randomness.
    seed = 42
    # Keep this fixed across runs so the same ordered input yields the same subset.
    # Training/validation splitting also remains fixed at 42 in realtime_trainer.py.
    sampling_seed = 42
    # Evaluation inputs must remain disjoint from all records used for fitting
    # and model selection. Missing files or overlapping content stop training.
    independent_test_files = {
        "nmi_test": os.path.join(os.path.dirname(__file__), "train_data_nmi", "mixed", "eval_test.fasta"),
        "nmi_species": os.path.join(os.path.dirname(__file__), "train_data_nmi", "mixed", "eval_generalization_species.fasta"),
        "nmi_family": os.path.join(os.path.dirname(__file__), "train_data_nmi", "mixed", "eval_generalization_family.fasta"),
    }
    
    # ========== Nucleotide Transformer v3 Core Configuration ==========
    transformer_model_repo = "Model/NTv3_8M_pre"  # NTv3 model repository (local snapshot; offline)
    embedding_dim = 256  # Output embedding dimension
    max_seq_len = 2048  # NTv3 maximum input length
    freeze_transformer = True  # Freeze NTv3
    unfreeze_modules = ["deconv_tower_blocks"]  # New: Only unfreeze this module
    trust_remote_code = True  # NTv3 requires trusting remote code
    
    # ========== Mamba Model Configuration ==========
    d_model = 256  # Model internal dimension, same as embedding_dim
    n_layer = 2  # Number of Mamba layers
    projection_dim = 128  # Contrastive learning projection dimension
    num_classes = 1  # Number of classification classes
    dropout_rate = 0.2  # Dropout rate
    
    # Mamba block configuration
    # block_type naming (architecture-overhaul, 2026-09-18):
    #   dual_path             = shared Mamba2 core (original)
    #   dual_path_transformer = shared TransformerEncoderLayer core (no mamba_ssm needed)
    #   dual_path_conv        = shared dilated-Conv (TCN) core (no mamba_ssm needed)
    #   future variants       = dual_path_<core>, e.g. dual_path_lstm
    block_type = "dual_path"
    # Classification head: "mlp" (legacy ours) or "linear" (baseline-matched
    # head-swap control; it can also be set without touching this file)
    classifier_head = "mlp"
    use_path_selection = True  # Whether to use path selection network
    path_selection_weight = 0.1  # Path selection loss weight
    
    # Enhancement module configuration
    use_local_global_attn = True  # Use local-global attention
    use_global_invariance = True  # Use global invariance encoding
    attn_num_heads = 4  # Number of attention heads
    # num_segments = 8  # Number of segments
    use_flash_attention = True  # Use Flash Attention
    
    # ========== Training Optimization Configuration ==========
    batch_size = 16  # Batch size (NTv3 requires large memory, 2048 sequence length needs smaller batch) ⚠️ Changed to 4
    learning_rate = 2e-5  # Base learning rate
    ntv3_learning_rate_multiplier = 0.1  # NTv3 parameter learning rate multiplier (1.0 learning rate)
    num_epochs = 20  # Number of training epochs
    weight_decay = 1e-5  # Weight decay
    grad_clip = 1.0  # Gradient clipping
    
    # Learning rate scheduler
    scheduler_type = "onecycle"
    onecycle_max_lr = 2e-5  # Reduce maximum learning rate
    onecycle_pct_start = 0.1  # Ramp-up phase proportion
    steps_per_epoch = 0  # 0 = auto: use the actual number of train batches per epoch
    
    # Early stopping configuration
    early_stop_patience = 5  # Early stopping patience value
    
    # Mixed precision training
    use_amp = True  # Use mixed precision training
    
    # ========== Variant Generation Configuration ==========
    # Semantic variant parameters
    conservative_energy_tolerance = 1.0
    min_pos_base_diff = 0.2
    max_pos_base_diff = 0.3
    pos_conservative_ratio = 0.2
    pos_syn_ratio = 0.8
    max_replace_ratio = 0.6
    max_semantic_attempts = 1000
    
    # DNA confusion variant parameters
    dna_confusion_min_fragment_len = 20
    dna_confusion_max_fragment_len = 200
    dna_confusion_min_fragments = 5
    dna_confusion_max_fragments = 10
    dna_confusion_flip_ratio = 0.3
    dna_confusion_max_attempts = 100
    
    # Negative sample configuration
    negative_mutation_rate = 0.5
    negative_max_attempts = 20
    
    # Number of positive views
    num_positive_views = 2  # Semantic similarity + DNA confusion
    
    # ========== Loss Configuration ==========
    contrastive_weight = 0.3  # Contrastive loss weight
    variant_specialization_weight = 0.2  # Alignment loss weight
    triplet_margin = 0.2  # Triplet loss margin
    triplet_temperature = 0.1  # Temperature parameter
    use_hard_triplet = False  # Whether to use hard triplets
    
    # ========== System Configuration ==========
    num_workers = min(os.cpu_count(), 16)  # Number of data loading workers
    pin_memory = True  # Memory pinning
    prefetch_factor = 2  # Prefetch factor
    persistent_workers = True  # Persistent workers
    
    # GPU optimization
    enable_tf32 = True  # Enable TF32 acceleration
    compile_model = False  # Compile model (optional)
    
    # Cache configuration
    use_caching = False  # Enable caching for better performance
    
    def __init__(self):
        """Initialize configuration.

        Importing the configuration must not touch the filesystem: containers
        may run as an unprivileged user and the working directory may be
        read-only. Run entry points create their output directory explicitly.
        """
        
        # The trainer initializes RNGs at the start of each run, after any
        # configuration overrides. Constructing a config must not reset them.
        
        # GPU optimization settings
        if self.enable_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Print configuration summary
        self.print_config_summary()

    def print_config_summary(self):
        """Print configuration summary"""
        print("=" * 70)
        print("🧬 Nucleotide Transformer v3 Mamba Configuration Summary (2048 length)")
        print("=" * 70)
        
        print(f"\n📊 Core Configuration:")
        print(f"  Model type: {self.model_type}")
        print(f"  Device: {self.device}")
        print(f"  Run mode: {self.run_mode}")
        print(f"  Output directory: {self.output_dir}")
        
        print(f"\n📈 Data Configuration:")
        print(f"  Maximum sequence length: {self.max_seq_length} ⚡")
        print(f"  Positive samples: {self.positive_fasta}")
        print(f"  Negative samples: {self.negative_fasta}")
        print(f"  Downsampling seed (fixed across runs): {self.sampling_seed}")
        
        print(f"\n🧬 Nucleotide Transformer v3:")
        print(f"  Model repository: {self.transformer_model_repo}")
        print(f"  Embedding dimension: {self.embedding_dim} (NTv3 hidden layer: 256)")
        print(f"  Maximum input length: {self.max_seq_len} ⚡")
        print(f"  Fully trainable: {not self.freeze_transformer} ⭐")
        print(f"  NTv3 learning rate: {self.learning_rate * self.ntv3_learning_rate_multiplier:.1e}")
        
        print(f"\n🤖 Mamba Model:")
        print(f"  Model dimension: {self.d_model}")
        print(f"  Number of layers: {self.n_layer}")
        print(f"  Block type: {self.block_type}")
        print(f"  Path selection: {self.use_path_selection}")
        print(f"  Local-global attention: {self.use_local_global_attn}")
        print(f"  Global invariance encoding: {self.use_global_invariance}")
        
        print(f"\n⚡ Training Configuration:")
        print(f"  Batch size: {self.batch_size} ⚠️ (2048 length requires small batch)")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Number of epochs: {self.num_epochs}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Early stopping patience: {self.early_stop_patience}")
        print(f"  Training seed (Python/NumPy/CPU/CUDA): {self.seed}")
        
        print(f"\n🎯 Loss Configuration:")
        print(f"  Contrastive loss weight: {self.contrastive_weight}")
        print(f"  Alignment loss weight: {self.variant_specialization_weight}")
        print(f"  Path selection loss weight: {self.path_selection_weight}")
        
        print(f"\n🔄 Variant Generation:")
        print(f"  Semantic variants: {self.max_semantic_attempts} attempts")
        print(f"  DNA confusion: {self.dna_confusion_max_fragments} fragments")
        print(f"  Positive views: {self.num_positive_views} views")
        
        print(f"\n⚠️  Important Notes:")
        print(f"  - NTv3 hidden layer dimension: 256")
        print(f"  - Maximum sequence length: 2048")
        print(f"  - Batch size: {self.batch_size} (be careful with memory)")
        print(f"  - Cache disabled: ensuring feature dimension consistency")
        
        print("\n" + "=" * 70)

    def get_model_kwargs(self):
        """Get model initialization parameters - simplified version"""
        return {
            # NTv3 core parameters
            'transformer_model_repo': self.transformer_model_repo,
            'embedding_dim': self.embedding_dim,
            'max_seq_len': self.max_seq_len,
            'freeze_transformer': self.freeze_transformer,
            'trust_remote_code': self.trust_remote_code,
            'unfreeze_modules': self.unfreeze_modules,  # Pass unfreeze modules
            
            # Mamba model parameters
            'd_model': self.d_model,
            'n_layer': self.n_layer,
            'projection_dim': self.projection_dim,
            'num_classes': self.num_classes,
            'variant_specialization_weight': self.variant_specialization_weight,
            'block_type': self.block_type,
            'dropout_rate': self.dropout_rate,
            'use_path_selection': self.use_path_selection,
            'path_selection_weight': self.path_selection_weight,
            
            # Enhancement module parameters
            'use_local_global_attn': self.use_local_global_attn,
            'use_global_invariance': self.use_global_invariance,
            'attn_num_heads': self.attn_num_heads,
            # 'num_segments': self.num_segments,
            'use_flash_attention': self.use_flash_attention,
            
            # System parameters
            'use_caching': self.use_caching,
            'enable_tf32': self.enable_tf32,
            'compile_model': self.compile_model,
            'device': str(self.device)
        }

    def get_training_config(self):
        """Get training configuration"""
        return {
            'seed': self.seed,
            'sampling_seed': self.sampling_seed,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'ntv3_learning_rate_multiplier': self.ntv3_learning_rate_multiplier,
            'num_epochs': self.num_epochs,
            'weight_decay': self.weight_decay,
            'grad_clip': self.grad_clip,
            
            'contrastive_weight': self.contrastive_weight,
            'variant_specialization_weight': self.variant_specialization_weight,
            
            'use_amp': self.use_amp,
            'scheduler_type': self.scheduler_type,
            'onecycle_max_lr': self.onecycle_max_lr,
            'onecycle_pct_start': self.onecycle_pct_start,
            'steps_per_epoch': self.steps_per_epoch,
            'early_stop_patience': self.early_stop_patience,
        }

    def get_variant_generation_config(self):
        """Get variant generation configuration - simplified version"""
        return {
            # Semantic variant parameters
            'conservative_energy_tolerance': self.conservative_energy_tolerance,
            'min_pos_base_diff': self.min_pos_base_diff,
            'max_pos_base_diff': self.max_pos_base_diff,
            'pos_conservative_ratio': self.pos_conservative_ratio,
            'pos_syn_ratio': self.pos_syn_ratio,
            'max_replace_ratio': self.max_replace_ratio,
            'max_semantic_attempts': self.max_semantic_attempts,
            
            # DNA confusion parameters
            'min_fragment_len': self.dna_confusion_min_fragment_len,
            'max_fragment_len': self.dna_confusion_max_fragment_len,
            'min_fragments': self.dna_confusion_min_fragments,
            'max_fragments': self.dna_confusion_max_fragments,
            'flip_ratio': self.dna_confusion_flip_ratio,
            'max_confusion_attempts': self.dna_confusion_max_attempts,
            
            # Negative sample parameters
            'negative_mutation_rate': self.negative_mutation_rate,
            'max_negative_attempts': self.negative_max_attempts,
        }

    def get_loss_config(self):
        """Get loss configuration"""
        return {
            'contrastive_weight': self.contrastive_weight,
            'variant_specialization_weight': self.variant_specialization_weight,
            'triplet_margin': self.triplet_margin,
            'triplet_temperature': self.triplet_temperature,
            'use_hard_triplet': self.use_hard_triplet
        }

# Global configuration instance
config = Config()
