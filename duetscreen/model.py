import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any

# Real encoder
from .encoder import NTv3TrainableEmbedding as _RealEncoder

MAMBA_AVAILABLE = True
try:
    from mamba_ssm.modules.mamba2 import Mamba2
except ImportError:
    MAMBA_AVAILABLE = False

def _require_mamba(block_name: str):
    if not MAMBA_AVAILABLE:
        raise ImportError(f"{block_name} requires the mamba_ssm package")

# ============================================================================
#
#
#
#      -> (output[B,L,d], path_weights|None, path_loss|None)
#   2) NucleotideMambaModel.forward_sequence(sequences, variant_type, training_mode)
#      -> (class_pred[B,1], pooled[B,d], contrastive_feat[B,128], path_loss)
#   3) NucleotideMambaModel.forward(original_sequences, labels, semantic_sequences,
# ============================================================================

import torch
import torch.nn as nn


class SimplifiedLocalGlobalAttention(nn.Module):
    """Simplified local-global hierarchical attention module
    Removes local convolution, focuses on functional semantic feature capture
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_flash_attention = use_flash_attention
        
        # Check if Flash Attention is supported
        self.has_flash_attention = hasattr(F, 'scaled_dot_product_attention')
        
        # Multi-head attention mechanism
        if not self.has_flash_attention or not use_flash_attention:
            self.multihead_attn = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True
            )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # Output normalization
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward_flash_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Global attention using Flash Attention acceleration"""
        batch_size, seq_len, d_model = x.shape
        
        # Reshape for multi-head
        q = x.view(batch_size, seq_len, self.num_heads, d_model // self.num_heads).transpose(1, 2)
        k = x.view(batch_size, seq_len, self.num_heads, d_model // self.num_heads).transpose(1, 2)
        v = x.view(batch_size, seq_len, self.num_heads, d_model // self.num_heads).transpose(1, 2)
        
        # Create causal mask
        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), 
            diagonal=1
        )
        
        # Use Flash Attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        # Reshape back to original shape
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return attn_output
    
    def forward_standard_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Standard multi-head attention"""
        seq_len = x.shape[1]
        
        # Create causal mask
        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), 
            diagonal=1
        )
        
        attn_output, _ = self.multihead_attn(
            x, x, x, 
            attn_mask=attn_mask,
            need_weights=False
        )
        
        return attn_output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 1. Global multi-head attention
        if self.has_flash_attention and self.use_flash_attention:
            global_features = self.forward_flash_attention(x)
        else:
            global_features = self.forward_standard_attention(x)
        
        # 2. Gated fusion: balance original features and attention features
        gate_input = torch.cat([x, global_features], dim=-1)
        gate_weights = self.gate(gate_input)
        
        # 3. Weighted fusion
        combined = gate_weights * x + (1 - gate_weights) * global_features
        
        # 4. Normalization
        output = self.norm(combined)
        output = self.dropout(output)
        
        return output


class EfficientMultiScaleCrossSegmentAttention(nn.Module):
    """Efficient multi-scale cross-segment attention module
    Optimized segment splitting for improved training efficiency
    """
    
    def __init__(
        self,
        d_model: int,
        segment_lengths: List[int] = [4, 8, 16],
        dropout: float = 0.1,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.segment_lengths = segment_lengths
        self.use_flash_attention = use_flash_attention
        
        # Check if Flash Attention is supported
        self.has_flash_attention = hasattr(F, 'scaled_dot_product_attention')
        
        # Multi-scale attention modules
        self.segment_attentions = nn.ModuleList()
        for _ in segment_lengths:
            # Use linear layers for preprocessing to reduce dimension
            pre_linear = nn.Linear(d_model, d_model // 2)
            post_linear = nn.Linear(d_model // 2, d_model)
            
            # Cross-segment attention
            if self.has_flash_attention and use_flash_attention:
                # For Flash Attention, use custom implementation
                cross_attn = EfficientCrossSegmentAttention(d_model // 2, dropout=dropout)
            else:
                cross_attn = nn.MultiheadAttention(
                    d_model // 2, num_heads=4, dropout=dropout, batch_first=True
                )
            
            self.segment_attentions.append(
                nn.ModuleDict({
                    'pre_linear': pre_linear,
                    'attention': cross_attn,
                    'post_linear': post_linear
                })
            )
        
        # Multi-scale feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * len(segment_lengths), d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Register buffer for precomputed segment indices (optimization)
        self.register_buffer('segment_indices', None, persistent=False)
        
    def compute_efficient_segments(self, x: torch.Tensor, segment_len: int) -> torch.Tensor:
        """Efficiently compute segment features, avoiding repeated reshape operations"""
        batch_size, seq_len, d_model = x.shape
        
        # Calculate number of segments
        num_segments = (seq_len + segment_len - 1) // segment_len
        
        # Use unfold for efficient segmented pooling
        if seq_len % segment_len == 0:
            # If perfectly divisible, reshape directly
            segments = x.view(batch_size, num_segments, segment_len, d_model)
        else:
            # Otherwise use padding + reshape
            new_seq_len = num_segments * segment_len
            padding_size = new_seq_len - seq_len
            
            if padding_size > 0:
                # Pad only the last dimension
                x_padded = F.pad(x, (0, 0, 0, padding_size))
            else:
                x_padded = x
            
            segments = x_padded.view(batch_size, num_segments, segment_len, d_model)
        
        # Average pooling to get segment-level features
        segment_features = segments.mean(dim=2)  # [batch_size, num_segments, d_model]
        
        return segment_features, num_segments
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        multi_scale_features = []
        
        # Apply cross-segment attention for each scale
        for i, segment_len in enumerate(self.segment_lengths):
            # 1. Preprocessing dimensionality reduction
            x_pre = self.segment_attentions[i]['pre_linear'](x)
            
            # 2. Efficiently compute segment features
            segment_features, num_segments = self.compute_efficient_segments(x_pre, segment_len)
            
            # 3. Cross-segment attention
            if self.has_flash_attention and self.use_flash_attention:
                segment_features_attn = self.segment_attentions[i]['attention'](segment_features)
            else:
                segment_features_attn, _ = self.segment_attentions[i]['attention'](
                    segment_features, segment_features, segment_features
                )
            
            # 4. Post-processing dimensionality increase
            segment_features_attn = self.segment_attentions[i]['post_linear'](segment_features_attn)
            
            # 5. Broadcast segment features back to original sequence length
            segment_len_actual = (seq_len + num_segments - 1) // num_segments
            segment_features_expanded = segment_features_attn.repeat_interleave(
                segment_len_actual, dim=1
            )
            
            # 6. Truncate to original length
            segment_features_expanded = segment_features_expanded[:, :seq_len, :]
            
            multi_scale_features.append(segment_features_expanded)
        
        # Multi-scale feature fusion
        if len(multi_scale_features) > 1:
            fused = torch.cat(multi_scale_features, dim=-1)
            output = self.fusion(fused)
        else:
            output = multi_scale_features[0]
        
        output = self.dropout(output)
        
        return output


class EfficientCrossSegmentAttention(nn.Module):
    """Efficient cross-segment attention module with Flash Attention support"""
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = dropout
        
        # Check if Flash Attention is supported
        self.has_flash_attention = hasattr(F, 'scaled_dot_product_attention')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_segments, d_model = x.shape
        
        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        if self.has_flash_attention:
            # Reshape for multi-head
            q = q.view(batch_size, num_segments, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, num_segments, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, num_segments, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Use Flash Attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.dropout if self.training else 0.0
            )
            
            # Reshape back to original shape
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_segments, d_model)
        else:
            # Standard multi-head attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0
            )
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class AdaptiveMultiScaleAttention(nn.Module):
    """Adaptive multi-scale attention module
    Dynamically selects the optimal scale for improved efficiency
    """
    
    def __init__(
        self,
        d_model: int,
        max_scales: int = 3,
        dropout: float = 0.1,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.max_scales = max_scales
        
        # Scale selection gating
        self.scale_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, max_scales),
            nn.Softmax(dim=-1)
        )
        
        # Main attention module (using medium scale)
        self.main_attention = EfficientMultiScaleCrossSegmentAttention(
            d_model=d_model,
            segment_lengths=[8],  # Default to medium scale
            dropout=dropout,
            use_flash_attention=use_flash_attention
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Calculate scale selection weights
        pooled = x.mean(dim=1)  # [batch_size, d_model]
        scale_weights = self.scale_gate(pooled)  # [batch_size, max_scales]
        
        # Get main scale output
        main_output = self.main_attention(x)
        
        # Adaptive adjustment based on sequence length
        if seq_len < 16:
            # Short sequences use simple processing
            output = main_output
        elif seq_len < 64:
            # Medium sequences
            output = main_output * 0.8 + x * 0.2
        else:
            # Long sequences use stronger attention
            output = main_output
        
        output = self.dropout(output)
        
        return output


class BasicMambaBlock(nn.Module):
    """Basic Mamba block (without residual connection)"""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        _require_mamba("BasicMambaBlock")
        self.d_model = d_model
        
        # Ensure d_conv is between 2 and 4
        d_conv = max(2, min(4, d_conv))
        
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x


class ConvNeXt1DBranch(nn.Module):
    """Shape-preserving correction branch, adapted from the ConvNeXt idea.

    Pre-LN -> depthwise Conv1d -> Linear(d, 2d) -> GELU -> Linear(2d, d).
    Unlike the original 2-D ConvNeXt block, this adapter uses pre-normalization
    and 2x expansion. It returns ONLY a correction: no identity addition,
    LayerScale, DropPath, extra dropout, or routing loss is hidden inside.
    Reference: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, d_model: int, kernel_size: int = 7, expansion: int = 2):
        super().__init__()
        if d_model <= 0 or expansion <= 0:
            raise ValueError("d_model and expansion must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.norm = nn.LayerNorm(d_model)
        self.dwconv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2, groups=d_model,
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Linear(expansion * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        correction = self.norm(x)
        correction = self.dwconv(correction.transpose(1, 2)).transpose(1, 2)
        return self.channel_mlp(correction)


class ParallelMambaConvNeXtBlock(BasicMambaBlock):
    """Two parallel transforms of the SAME input, with zero-start fusion.

    Main-path norm/mamba/dropout and their state-dict names are inherited
    unchanged from BasicMambaBlock. In eval mode, with matching main-path
    weights and alpha=0, this block is exactly the basic block. This does not
    guarantee equal training trajectories or a non-decreasing test score.

    The frozen block contract is (output, path_weights, path_loss). There is
    no sample-dependent routing or path supervision, so the last two are None.
    """

    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4,
        expand: int = 2, layer_idx: int = 0, use_path_selection: bool = False,
    ):
        super().__init__(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.layer_idx = layer_idx
        # Accepted for constructor compatibility, intentionally never enabled.
        self.use_path_selection = False
        self.aux_branch = ConvNeXt1DBranch(d_model, kernel_size=7, expansion=2)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(
        self, x: torch.Tensor, variant_type: Optional[str] = None,
        training_mode: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        main = super().forward(x)
        # Use X, not main. Do not skip this computation when alpha == 0:
        # alpha still needs the normally initialized correction for its gradient.
        correction = self.aux_branch(x)
        output = main + self.alpha * correction
        return output, None, None


class ReadoutMambaBlock(BasicMambaBlock):
    """Unchanged basic computation with the registered-block tuple contract.

    The second path is a network-level readout of X, not a second per-block
    operator. State-dict keys of the basic main path are preserved.
    """

    def __init__(self, d_model: int, layer_idx: int = 0,
                 use_path_selection: bool = False):
        super().__init__(d_model=d_model)
        self.layer_idx = layer_idx
        self.use_path_selection = False

    def forward(self, x: torch.Tensor, variant_type: Optional[str] = None,
                training_mode: bool = True):
        return super().forward(x), None, None


def _validate_pool_input(x: torch.Tensor, valid_mask: Optional[torch.Tensor]):
    if x.ndim != 3 or x.shape[1] == 0:
        raise ValueError("Pooling requires nonempty [B,L,d] features")
    if valid_mask is not None:
        if valid_mask.shape != x.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B,L], True for valid tokens")
        if valid_mask.device != x.device:
            raise ValueError("valid_mask and features must be on the same device")
        if not valid_mask.any(dim=1).all().item():
            raise ValueError("Each sample must have at least one valid token")


def _masked_token_mean(x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
    _validate_pool_input(x, valid_mask)
    # FP32 accumulation for AMP inputs; preserve double precision when requested.
    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        values = x.to(work_dtype)
        if valid_mask is None:
            result = values.mean(dim=1)
        else:
            values = values.masked_fill(~valid_mask.unsqueeze(-1), 0)
            result = values.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True)
    return result.to(x.dtype)


class TokenAttentionPool(nn.Module):
    """Single-query attention pooling, inspired by Ilse et al. (ICML 2018).

    score_i = w^T tanh(W LN(x_i)); output = sum_i softmax(score)_i * x_i.
    This is the ungated attention-pooling idea, not a full MIL classifier.
    LN is used only for scores; values remain the input features. Complexity
    is linear in length. No L-by-L attention matrix or view-specific routing.
    """

    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0:
            raise ValueError("d_model and hidden_dim must be positive")
        self.score_norm = nn.LayerNorm(d_model)
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.Tanh(),
            # A shared score bias cancels in softmax, so omit it.
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None,
                return_weights: bool = False):
        _validate_pool_input(x, valid_mask)
        # Also remove masked values BEFORE scoring, so masked padding cannot
        # contaminate parameter gradients (including when it contains NaNs).
        values = x if valid_mask is None else x.masked_fill(~valid_mask.unsqueeze(-1), 0)
        logits = self.score(self.score_norm(values)).squeeze(-1)
        work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = logits.to(work_dtype)
            if valid_mask is not None:
                logits = logits.masked_fill(~valid_mask, float('-inf'))
            weights = torch.softmax(logits, dim=1)
            pooled = torch.bmm(weights.unsqueeze(1), values.to(work_dtype)).squeeze(1)
        pooled = pooled.to(x.dtype)
        return (pooled, weights) if return_weights else pooled


class LowRankReadoutRefinement(nn.Module):
    """Two equally parameterized refinements of the base fusion.

    mlp:       O(tanh(R [u;v]))
    bilinear:  O(tanh(A u) * tanh(B v))

    Each uses exactly 3*d_model*rank weights; no biases, dropout, scalar
    gates or zero-initialized output matrices. 'bilinear' is a nonlinear
    adaptation of factorized Hadamard fusion, not a strictly bilinear map
    after tanh. Its products are in learned coordinates, not arbitrary
    same-index encoder/Mamba channels.
    """

    def __init__(self, d_model: int, rank: int = 64, mode: str = 'mlp'):
        super().__init__()
        if d_model <= 0 or rank <= 0:
            raise ValueError("d_model and fusion rank must be positive")
        if mode not in ('mlp', 'bilinear'):
            raise ValueError("Refinement mode must be 'mlp' or 'bilinear'")
        self.mode = mode
        self.rank = rank
        if mode == 'mlp':
            self.joint_projection = nn.Linear(2 * d_model, rank, bias=False)
        else:
            self.main_projection = nn.Linear(d_model, rank, bias=False)
            self.direct_projection = nn.Linear(d_model, rank, bias=False)
        self.output = nn.Linear(rank, d_model, bias=False)

    def forward(self, main: torch.Tensor, direct: torch.Tensor):
        if self.mode == 'mlp':
            hidden = torch.tanh(self.joint_projection(torch.cat((main, direct), dim=-1)))
        else:
            hidden = torch.tanh(self.main_projection(main)) * torch.tanh(self.direct_projection(direct))
        return self.output(hidden)


class DualReadoutPool(nn.Module):
    """Late fusion of mean(stack(X)) and a direct summary of input X.

    Both summaries are normalized and retain the base concat/Linear
    term. Optional v4 refinement acts only on these two [B,d] summaries,
    before the original output norm and heads. All added matrices use
    ordinary nonzero initialization. Useful complementarity is NOT guaranteed.

    direct_pool='mean' is the direct-access control. direct_source='stack'
    is an optional same-level readout control, not the default parallel path.
    """

    def __init__(self, d_model: int, hidden_dim: int = 64,
                 direct_pool: str = 'attention', direct_source: str = 'input',
                 fusion_mode: str = 'linear', fusion_rank: int = 64):
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0:
            raise ValueError("d_model and hidden_dim must be positive")
        if direct_pool not in ('attention', 'mean'):
            raise ValueError("direct_pool must be 'attention' or 'mean'")
        if direct_source not in ('input', 'stack'):
            raise ValueError("direct_source must be 'input' or 'stack'")
        if fusion_mode not in ('linear', 'mlp', 'bilinear'):
            raise ValueError("fusion_mode must be 'linear', 'mlp' or 'bilinear'")
        self.d_model = d_model
        self.direct_pool = direct_pool
        self.direct_source = direct_source
        self.fusion_mode = fusion_mode
        self.attention = TokenAttentionPool(d_model, hidden_dim) if direct_pool == 'attention' else None
        self.main_norm = nn.LayerNorm(d_model)
        self.direct_norm = nn.LayerNorm(d_model)
        self.fusion = nn.Linear(2 * d_model, d_model, bias=False)
        self.refinement = (None if fusion_mode == 'linear' else
                           LowRankReadoutRefinement(d_model, fusion_rank, fusion_mode))

    def summaries(self, inputs: torch.Tensor, hidden: torch.Tensor,
                  valid_mask: Optional[torch.Tensor] = None):
        if inputs.shape != hidden.shape or inputs.shape[-1] != self.d_model:
            raise ValueError("Both readout paths require matching [B,L,d_model] features")
        main = self.main_norm(_masked_token_mean(hidden, valid_mask))
        source = inputs if self.direct_source == 'input' else hidden
        if self.attention is not None:
            direct, weights = self.attention(source, valid_mask, return_weights=True)
        else:
            direct = _masked_token_mean(source, valid_mask)
            weights = torch.ones(source.shape[:2], device=source.device, dtype=torch.float32)
            if valid_mask is not None:
                weights = weights.masked_fill(~valid_mask, 0)
            weights = weights / weights.sum(dim=1, keepdim=True)
        return main, self.direct_norm(direct), weights

    def fuse(self, main: torch.Tensor, direct: torch.Tensor,
             include_refinement: bool = True):
        base = self.fusion(torch.cat((main, direct), dim=-1))
        if include_refinement and self.refinement is not None:
            return base + self.refinement(main, direct)
        return base

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None):
        main, direct, _ = self.summaries(inputs, hidden, valid_mask)
        return self.fuse(main, direct)






class BasicTransformerBlock(nn.Module):
    """Vanilla stack block with a pre-norm TransformerEncoderLayer core.

    Core-swap control on the simplified stack (control specification priority 2):
    same LN -> core -> Dropout shape as BasicMambaBlock; only the
    sequence-mixing core differs.
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.layer(x)
        return self.dropout(x)


class BasicConvBlock(nn.Module):
    """Vanilla stack block with the shared dilated-Conv core (core-swap control)."""

    def __init__(self, d_model: int, width_multiplier: int = 2,
                 kernel_size: int = 5, dilations: Tuple[int, int] = (1, 2),
                 dropout: float = 0.1):
        super().__init__()
        self.core = _ConvCore(d_model=d_model, width_multiplier=width_multiplier,
                              kernel_size=kernel_size, dilations=dilations,
                              dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)


class ReadoutConvBlock(BasicConvBlock):
    """Unchanged proven Conv core, with the registered-block tuple contract.

    The complete independent Conv stack is used only by the conv variant. There is no
    shared Mamba, alpha, routing, or per-layer exchange with the main stack.
    """

    def __init__(self, d_model: int, layer_idx: int = 0,
                 use_path_selection: bool = False):
        super().__init__(d_model=d_model)
        self.layer_idx = layer_idx
        self.use_path_selection = False

    def forward(self, x: torch.Tensor, variant_type: Optional[str] = None,
                training_mode: bool = True):
        return super().forward(x), None, None


class EnhancedMambaBlock(nn.Module):
    """Enhanced Mamba block - enhanced features directly as Mamba input"""
    
    def __init__(
        self, 
        d_model: int, 
        d_state: int = 16, 
        d_conv: int = 4,
        expand: int = 2,
        use_local_global_attn: bool = True,
        use_global_invariance: bool = True,
        attn_num_heads: int = 4,
        use_flash_attention: bool = True,
        use_adaptive_scales: bool = True  # New: whether to use adaptive scales
    ):
        super().__init__()
        _require_mamba("EnhancedMambaBlock")
        self.d_model = d_model
        self.use_local_global_attn = use_local_global_attn
        self.use_global_invariance = use_global_invariance

        # Ensure d_conv is between 2 and 4
        d_conv = max(2, min(4, d_conv))

        # Enhancement modules
        if use_local_global_attn:
            self.local_global_attn = SimplifiedLocalGlobalAttention(
                d_model=d_model,
                num_heads=attn_num_heads,
                use_flash_attention=use_flash_attention
            )
        else:
            self.local_global_attn = None
        
        if use_global_invariance:
            if use_adaptive_scales:
                self.global_invariance_encoder = AdaptiveMultiScaleAttention(
                    d_model=d_model,
                    use_flash_attention=use_flash_attention
                )
            else:
                self.global_invariance_encoder = EfficientMultiScaleCrossSegmentAttention(
                    d_model=d_model,
                    segment_lengths=[4, 8, 16],
                    use_flash_attention=use_flash_attention
                )
        else:
            self.global_invariance_encoder = None
        
        # Mamba block
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply enhancement modules
        if self.local_global_attn is not None:
            x = self.local_global_attn(x)
        
        if self.global_invariance_encoder is not None:
            x = self.global_invariance_encoder(x)
        
        # Mamba processing
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        
        return x


class DualPathMambaBlock(nn.Module):
    """Dual-path Mamba block - each path uses different enhanced features, shares Mamba parameters"""
    
    def __init__(
        self, 
        d_model: int, 
        d_state: int = 16, 
        d_conv: int = 4,
        expand: int = 2,
        layer_idx: int = 0,
        use_path_selection: bool = True,
        path_selection_strategy: str = "adaptive_learning",
        path_diversity_weight: float = 0.1,
        path_sparsity_weight: float = 0.05,
        # Enhancement module parameters
        use_local_global_attn: bool = True,
        use_global_invariance: bool = True,
        attn_num_heads: int = 4,
        use_flash_attention: bool = True,
        use_adaptive_scales: bool = True,  # New: whether to use adaptive scales
        # T1 control (control specification): zero every INTERNAL dropout site
        # (attention internals, gate network, fusion MLP) while keeping the
        # block-output Dropout at its historical 0.1.
        zero_internal_dropout: bool = False
    ):
        super().__init__()
        _require_mamba("DualPathMambaBlock")
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.use_path_selection = use_path_selection
        self.path_selection_strategy = path_selection_strategy
        self.path_diversity_weight = path_diversity_weight
        self.path_sparsity_weight = path_sparsity_weight
        self.use_local_global_attn = use_local_global_attn
        self.use_global_invariance = use_global_invariance
        
        # Ensure d_conv is between 2 and 4
        d_conv = max(2, min(4, d_conv))
        
        # Path 1: Simplified local-global attention enhancement (focusing on functional semantics)
        self.path1_enhance = SimplifiedLocalGlobalAttention(
            d_model=d_model,
            num_heads=attn_num_heads,
            use_flash_attention=use_flash_attention
        ) if use_local_global_attn else nn.Identity()
        
        self.path1_norm = nn.LayerNorm(d_model)
        
        # Path 2: Efficient multi-scale cross-segment attention enhancement (focusing on attack strategies)
        if use_global_invariance:
            if use_adaptive_scales:
                self.path2_enhance = AdaptiveMultiScaleAttention(
                    d_model=d_model,
                    use_flash_attention=use_flash_attention
                )
            else:
                self.path2_enhance = EfficientMultiScaleCrossSegmentAttention(
                    d_model=d_model,
                    segment_lengths=[4, 8, 16],
                    use_flash_attention=use_flash_attention
                )
        else:
            self.path2_enhance = nn.Identity()
        
        self.path2_norm = nn.LayerNorm(d_model)
        
        # Shared Mamba module
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Path selection gating network
        if use_path_selection:
            # Use summary of enhanced features as input
            gate_input_dim = d_model * 2  # Summary of enhanced features from both paths
            
            self.gate_network = nn.Sequential(
                nn.Linear(gate_input_dim, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, 2),
                nn.Softmax(dim=-1)
            )
        
        # Output fusion layer
        self.output_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.dropout = nn.Dropout(0.1)

        if zero_internal_dropout:
            for name, mod in self.named_modules():
                if mod is self.dropout:
                    continue
                if isinstance(mod, nn.Dropout):
                    mod.p = 0.0
    
    def compute_path_selection_loss(self, learned_weights: torch.Tensor, 
                                   variant_type: Optional[str] = None) -> torch.Tensor:
        """Calculate path selection loss"""
        if variant_type is not None:
            # Use variant type as soft supervision
            if variant_type == "confusion":
                target_bias = torch.tensor([[0.3, 0.7]], device=learned_weights.device)
            else:  # "original" or "semantic"
                target_bias = torch.tensor([[0.7, 0.3]], device=learned_weights.device)
            
            batch_size = learned_weights.shape[0]
            target = target_bias.repeat(batch_size, 1)
            
            # KL divergence loss (diversity)
            diversity_loss = F.kl_div(
                F.log_softmax(learned_weights, dim=-1),
                target,
                reduction='batchmean'
            )
            
            # Sparsity loss (entropy minimization)
            eps = 1e-8
            entropy = - (learned_weights * torch.log(learned_weights + eps)).sum(dim=-1)
            sparsity_loss = entropy.mean()
            
            # Combined loss
            path_selection_loss = (
                self.path_diversity_weight * diversity_loss +
                self.path_sparsity_weight * sparsity_loss
            )
        else:
            # Use only sparsity loss
            eps = 1e-8
            entropy = - (learned_weights * torch.log(learned_weights + eps)).sum(dim=-1)
            sparsity_loss = entropy.mean()
            path_selection_loss = self.path_sparsity_weight * sparsity_loss
        
        return path_selection_loss
    
    def forward(
        self, 
        x: torch.Tensor, 
        variant_type: Optional[str] = None,
        training_mode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        
        # 1. Generate enhanced features
        path1_enhanced = self.path1_enhance(x)
        path2_enhanced = self.path2_enhance(x)
        
        # 2. Path selection
        path_selection_loss = None
        if self.use_path_selection:
            # Use summary of enhanced features as gate input
            gate_input = torch.cat([
                path1_enhanced.mean(dim=1),
                path2_enhanced.mean(dim=1)
            ], dim=-1)
            
            learned_weights = self.gate_network(gate_input)  # [batch_size, 2]
            
            if training_mode:
                path_selection_loss = self.compute_path_selection_loss(
                    learned_weights, variant_type
                )
            
            path_weights = learned_weights
        else:
            # Fixed weights
            if variant_type == "confusion":
                path_weights = torch.tensor([[0.3, 0.7]], device=x.device).repeat(batch_size, 1)
            else:
                path_weights = torch.tensor([[0.7, 0.3]], device=x.device).repeat(batch_size, 1)
        
        # 3. Process enhanced features with shared Mamba
        # Path 1: Functional semantic features -> Shared Mamba
        path1_out = self.mamba(self.path1_norm(path1_enhanced))
        
        # Path 2: Attack response features -> Shared Mamba
        path2_out = self.mamba(self.path2_norm(path2_enhanced))
        
        # 4. Dual-path output fusion
        # 4.1 Weighted fusion based on path weights
        path_weights_expanded = path_weights.unsqueeze(1).unsqueeze(-1)
        path_outputs = torch.stack([path1_out, path2_out], dim=2)
        weighted_output = (path_outputs * path_weights_expanded).sum(dim=2)
        
        # 4.2 Optional feature fusion layer
        combined_features = torch.cat([path1_out, path2_out], dim=-1)
        fused_features = self.output_fusion(combined_features)
        
        # 4.3 Final output: 70% weighted output + 30% fused output
        output = 0.7 * weighted_output + 0.3 * fused_features
        output = self.dropout(output)
        
        return output, path_weights, path_selection_loss


class DualPathTransformerBlock(nn.Module):
    """Dual-path block with a shared Transformer core (architecture-overhaul variant).

    Interface-compatible with DualPathMambaBlock: identical path enhancement
    (semantic path + adversarial path), gating, path-selection losses and
    output fusion. The only change is the shared sequence-mixing core —
    a pre-norm TransformerEncoderLayer replaces Mamba2, so the experiment
    isolates the choice of sequence mixer while everything else stays fixed.
    Runs on hosts without mamba_ssm.
    """

    def __init__(
        self,
        d_model: int,
        layer_idx: int = 0,
        use_path_selection: bool = True,
        path_selection_strategy: str = "adaptive_learning",
        path_diversity_weight: float = 0.1,
        path_sparsity_weight: float = 0.05,
        # Enhancement module parameters
        use_local_global_attn: bool = True,
        use_global_invariance: bool = True,
        attn_num_heads: int = 4,
        use_flash_attention: bool = True,
        use_adaptive_scales: bool = True,
        # Transformer core parameters
        feedforward_multiplier: int = 4,
        core_dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.use_path_selection = use_path_selection
        self.path_selection_strategy = path_selection_strategy
        self.path_diversity_weight = path_diversity_weight
        self.path_sparsity_weight = path_sparsity_weight
        self.use_local_global_attn = use_local_global_attn
        self.use_global_invariance = use_global_invariance

        # Path 1: Simplified local-global attention enhancement (functional semantics)
        self.path1_enhance = SimplifiedLocalGlobalAttention(
            d_model=d_model,
            num_heads=attn_num_heads,
            use_flash_attention=use_flash_attention
        ) if use_local_global_attn else nn.Identity()

        self.path1_norm = nn.LayerNorm(d_model)

        # Path 2: Efficient multi-scale cross-segment attention enhancement (attack strategies)
        if use_global_invariance:
            if use_adaptive_scales:
                self.path2_enhance = AdaptiveMultiScaleAttention(
                    d_model=d_model,
                    use_flash_attention=use_flash_attention
                )
            else:
                self.path2_enhance = EfficientMultiScaleCrossSegmentAttention(
                    d_model=d_model,
                    segment_lengths=[4, 8, 16],
                    use_flash_attention=use_flash_attention
                )
        else:
            self.path2_enhance = nn.Identity()

        self.path2_norm = nn.LayerNorm(d_model)

        # Shared Transformer core (replaces the shared Mamba2 of DualPathMambaBlock)
        self.core = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=attn_num_heads,
            dim_feedforward=feedforward_multiplier * d_model,
            dropout=core_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # Path selection gating network (identical to the Mamba variant)
        if use_path_selection:
            gate_input_dim = d_model * 2
            self.gate_network = nn.Sequential(
                nn.Linear(gate_input_dim, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, 2),
                nn.Softmax(dim=-1)
            )

        # Output fusion layer (identical to the Mamba variant)
        self.output_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.dropout = nn.Dropout(0.1)

    compute_path_selection_loss = DualPathMambaBlock.compute_path_selection_loss

    def forward(
        self,
        x: torch.Tensor,
        variant_type: Optional[str] = None,
        training_mode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        # 1. Generate enhanced features
        path1_enhanced = self.path1_enhance(x)
        path2_enhanced = self.path2_enhance(x)

        # 2. Path selection
        path_selection_loss = None
        if self.use_path_selection:
            gate_input = torch.cat([
                path1_enhanced.mean(dim=1),
                path2_enhanced.mean(dim=1)
            ], dim=-1)

            learned_weights = self.gate_network(gate_input)

            if training_mode:
                path_selection_loss = self.compute_path_selection_loss(
                    learned_weights, variant_type
                )

            path_weights = learned_weights
        else:
            if variant_type == "confusion":
                path_weights = torch.tensor([[0.3, 0.7]], device=x.device).repeat(batch_size, 1)
            else:
                path_weights = torch.tensor([[0.7, 0.3]], device=x.device).repeat(batch_size, 1)

        # 3. Process enhanced features with the shared Transformer core
        path1_out = self.core(self.path1_norm(path1_enhanced))
        path2_out = self.core(self.path2_norm(path2_enhanced))

        # 4. Dual-path output fusion (identical to the Mamba variant)
        path_weights_expanded = path_weights.unsqueeze(1).unsqueeze(-1)
        path_outputs = torch.stack([path1_out, path2_out], dim=2)
        weighted_output = (path_outputs * path_weights_expanded).sum(dim=2)

        combined_features = torch.cat([path1_out, path2_out], dim=-1)
        fused_features = self.output_fusion(combined_features)

        output = 0.7 * weighted_output + 0.3 * fused_features
        output = self.dropout(output)

        return output, path_weights, path_selection_loss


class _ConvCore(nn.Module):
    """Stacked dilated 1-D convolutions with entry norm and residual.

    Sequence-mixing core for the architecture-overhaul variants: a small TCN
    style block (Conv k=5 dil=1 -> GELU -> Conv k=5 dil=2, wide bottleneck,
    pre-norm, residual). Same [B, L, D] -> [B, L, D] contract as the Mamba2 /
    TransformerEncoderLayer cores.
    """

    def __init__(self, d_model: int, width_multiplier: int = 2,
                 kernel_size: int = 5, dilations: Tuple[int, int] = (1, 2),
                 dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        width = d_model * width_multiplier
        self.conv1 = nn.Conv1d(d_model, width, kernel_size,
                               padding=dilations[0] * (kernel_size - 1) // 2,
                               dilation=dilations[0])
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(width, d_model, kernel_size,
                               padding=dilations[1] * (kernel_size - 1) // 2,
                               dilation=dilations[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)              # [B, D, L]
        x = self.drop(self.act(self.conv1(x)))
        x = self.conv2(x)
        x = x.transpose(1, 2)              # [B, L, D]
        return x + residual


class DualPathConvBlock(nn.Module):
    """Dual-path block with a shared dilated-Conv core (architecture-overhaul variant).

    Interface-compatible with DualPathMambaBlock / DualPathTransformerBlock:
    identical path enhancement, gating, path-selection losses and output
    fusion; only the shared sequence-mixing core differs. No mamba_ssm needed.
    """

    def __init__(
        self,
        d_model: int,
        layer_idx: int = 0,
        use_path_selection: bool = True,
        path_selection_strategy: str = "adaptive_learning",
        path_diversity_weight: float = 0.1,
        path_sparsity_weight: float = 0.05,
        # Enhancement module parameters
        use_local_global_attn: bool = True,
        use_global_invariance: bool = True,
        attn_num_heads: int = 4,
        use_flash_attention: bool = True,
        use_adaptive_scales: bool = True,
        # Conv core parameters
        width_multiplier: int = 2,
        kernel_size: int = 5,
        dilations: Tuple[int, int] = (1, 2),
        core_dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.use_path_selection = use_path_selection
        self.path_selection_strategy = path_selection_strategy
        self.path_diversity_weight = path_diversity_weight
        self.path_sparsity_weight = path_sparsity_weight
        self.use_local_global_attn = use_local_global_attn
        self.use_global_invariance = use_global_invariance

        # Path 1: Simplified local-global attention enhancement (functional semantics)
        self.path1_enhance = SimplifiedLocalGlobalAttention(
            d_model=d_model,
            num_heads=attn_num_heads,
            use_flash_attention=use_flash_attention
        ) if use_local_global_attn else nn.Identity()

        self.path1_norm = nn.LayerNorm(d_model)

        # Path 2: multi-scale cross-segment attention enhancement (attack strategies)
        if use_global_invariance:
            if use_adaptive_scales:
                self.path2_enhance = AdaptiveMultiScaleAttention(
                    d_model=d_model,
                    use_flash_attention=use_flash_attention
                )
            else:
                self.path2_enhance = EfficientMultiScaleCrossSegmentAttention(
                    d_model=d_model,
                    segment_lengths=[4, 8, 16],
                    use_flash_attention=use_flash_attention
                )
        else:
            self.path2_enhance = nn.Identity()

        self.path2_norm = nn.LayerNorm(d_model)

        # Shared dilated-Conv core (replaces the shared Mamba2 / Transformer core)
        self.core = _ConvCore(
            d_model=d_model,
            width_multiplier=width_multiplier,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=core_dropout,
        )

        # Path selection gating network (identical to the Mamba variant)
        if use_path_selection:
            gate_input_dim = d_model * 2
            self.gate_network = nn.Sequential(
                nn.Linear(gate_input_dim, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, 2),
                nn.Softmax(dim=-1)
            )

        # Output fusion layer (identical to the Mamba variant)
        self.output_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.dropout = nn.Dropout(0.1)

    compute_path_selection_loss = DualPathMambaBlock.compute_path_selection_loss

    def forward(
        self,
        x: torch.Tensor,
        variant_type: Optional[str] = None,
        training_mode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        # 1. Generate enhanced features
        path1_enhanced = self.path1_enhance(x)
        path2_enhanced = self.path2_enhance(x)

        # 2. Path selection
        path_selection_loss = None
        if self.use_path_selection:
            gate_input = torch.cat([
                path1_enhanced.mean(dim=1),
                path2_enhanced.mean(dim=1)
            ], dim=-1)

            learned_weights = self.gate_network(gate_input)

            if training_mode:
                path_selection_loss = self.compute_path_selection_loss(
                    learned_weights, variant_type
                )

            path_weights = learned_weights
        else:
            if variant_type == "confusion":
                path_weights = torch.tensor([[0.3, 0.7]], device=x.device).repeat(batch_size, 1)
            else:
                path_weights = torch.tensor([[0.7, 0.3]], device=x.device).repeat(batch_size, 1)

        # 3. Process enhanced features with the shared Conv core
        path1_out = self.core(self.path1_norm(path1_enhanced))
        path2_out = self.core(self.path2_norm(path2_enhanced))

        # 4. Dual-path output fusion (identical to the Mamba variant)
        path_weights_expanded = path_weights.unsqueeze(1).unsqueeze(-1)
        path_outputs = torch.stack([path1_out, path2_out], dim=2)
        weighted_output = (path_outputs * path_weights_expanded).sum(dim=2)

        combined_features = torch.cat([path1_out, path2_out], dim=-1)
        fused_features = self.output_fusion(combined_features)

        output = 0.7 * weighted_output + 0.3 * fused_features
        output = self.dropout(output)

        return output, path_weights, path_selection_loss


class NucleotideMambaModel(nn.Module):
    """Fully trainable NTv3 enhanced Mamba sequence model without residual connections"""
    
    def __init__(
        self,
        transformer_model_repo: str = "InstaDeepAI/NTv3_8M_pre",
        embedding_dim: int = 256,
        # Mamba model configuration
        d_model: int = 256,
        n_layer: int = 4,
        projection_dim: int = 256,
        num_classes: int = 1,
        variant_specialization_weight: float = 0.2,
        block_type: str = "dual_path",
        dropout_rate: float = 0.1,
        use_path_selection: bool = True,
        path_selection_weight: float = 1.0,
        max_seq_len: int = 512,  # Number of token bases
        # Enhancement module configuration
        use_local_global_attn: bool = True,
        use_global_invariance: bool = True,
        attn_num_heads: int = 4,
        use_flash_attention: bool = True,
        use_adaptive_scales: bool = True,  # New: whether to use adaptive scales
        # Transformer configuration - Note: We do NOT freeze here!
        freeze_transformer: bool = True,  # Changed to False for full trainability
        use_caching: bool = False,  # Force disable caching to ensure feature dimension consistency
        trust_remote_code: bool = True,  # v3 requires this
        # Architecture configuration
        enable_tf32: bool = True,
        compile_model: bool = False,
        device: Optional[str] = None,
        # Classification head: "mlp" (legacy ours: Linear-GELU-Dropout-Linear
        # after output_norm) or "linear" (single Linear, matched to the
        # baseline head; head-swap arm 2026-09-20)
        classifier_head: str = "mlp",
        # T1 control flag: zero internal dropouts of dual-path Mamba blocks
        zero_internal_dropout: bool = False,
        # v3 network-level parallel readout; ignored by historical types.
        readout_pool: str = "attention",
        readout_source: str = "input",
        readout_hidden_dim: int = 64,
        readout_fusion: str = "linear",
        readout_rank: int = 64
    ):
        super().__init__()
        
        # Store parameters
        self.transformer_model_repo = transformer_model_repo
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.n_layer = n_layer
        self.projection_dim = projection_dim
        self.num_classes = num_classes
        self.variant_specialization_weight = variant_specialization_weight
        self.block_type = block_type
        self.dropout_rate = dropout_rate
        self.use_path_selection = use_path_selection
        self.path_selection_weight = path_selection_weight
        self.max_seq_len = max_seq_len
        self.use_local_global_attn = use_local_global_attn
        self.use_global_invariance = use_global_invariance
        self.use_adaptive_scales = use_adaptive_scales
        self.freeze_transformer = freeze_transformer  # Now set to False
        self.use_caching = use_caching
        self.trust_remote_code = trust_remote_code
        self.classifier_head = classifier_head
        self.zero_internal_dropout = zero_internal_dropout
        self.readout_pool = readout_pool
        self.readout_source = readout_source
        self.readout_hidden_dim = readout_hidden_dim
        self.readout_fusion = readout_fusion
        self.readout_rank = readout_rank
        self.has_independent_stack = block_type in ('dual_mamba_conv', 'bidirectional_mamba')

        if block_type in ("parallel_mamba_convnext", "dual_readout_mamba") or self.has_independent_stack:
            # Legacy path flags are accepted but have no effect on this block.
            self.use_path_selection = False
            self.use_local_global_attn = False
            self.use_global_invariance = False
            self.use_adaptive_scales = False
        if (block_type == "dual_readout_mamba" or self.has_independent_stack) and n_layer < 1:
            raise ValueError("Dual readout networks require at least one block per stack")
        if self.has_independent_stack and (readout_pool, readout_source, readout_fusion) != ('mean', 'input', 'linear'):
            raise ValueError("Independent stacks require readout_pool='mean', readout_source='input', readout_fusion='linear'; use the named factories")
        
        # Auto-select device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # GPU optimization: Enable TF32 acceleration
        if enable_tf32 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✅ TF32 acceleration enabled")
        
        print("Initializing downstream sequence network")
        print(f"  Transformer model: {transformer_model_repo}")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Model dimension: {d_model}")
        print(f"  Number of layers: {n_layer}")
        print(f"  Block type: {block_type}")
        print(f"  Classifier head: {classifier_head}")
        print(f"  Maximum sequence length: {max_seq_len}")
        print(f"  Adaptive scales: {self.use_adaptive_scales}")
        print("  Encoder freezing and optimizer groups are controlled by project integration")
        
        self.nucleotide_embedding = _RealEncoder(
            model_repo=transformer_model_repo,
            output_dim=embedding_dim,
            max_seq_len=max_seq_len,
            use_cache=False,
            device=str(self.device),
            trust_remote_code=trust_remote_code,
            freeze_transformer=freeze_transformer
        )
        
        # 2. Project from embedding_dim to d_model (if dimensions differ)
        if embedding_dim != d_model:
            self.embedding_projection = nn.Sequential(
                nn.Linear(embedding_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
        else:
            self.embedding_projection = nn.Identity()
        
        # 3. Mamba block sequence (without residual connections)
        self.mamba_blocks = nn.ModuleList()
        for i in range(n_layer):
            if block_type == "dual_readout_mamba" or self.has_independent_stack:
                block = ReadoutMambaBlock(d_model=d_model, layer_idx=i)
            elif block_type == "parallel_mamba_convnext":
                block = ParallelMambaConvNeXtBlock(d_model=d_model, layer_idx=i)
            elif block_type == "dual_path":
                block = DualPathMambaBlock(
                    d_model=d_model,
                    layer_idx=i,
                    use_path_selection=use_path_selection,
                    use_local_global_attn=use_local_global_attn,
                    use_global_invariance=use_global_invariance,
                    attn_num_heads=attn_num_heads,
                    use_flash_attention=use_flash_attention,
                    use_adaptive_scales=use_adaptive_scales,
                    zero_internal_dropout=zero_internal_dropout
                )
            elif block_type == "dual_path_transformer":
                # Architecture-overhaul variant: shared Transformer core
                block = DualPathTransformerBlock(
                    d_model=d_model,
                    layer_idx=i,
                    use_path_selection=use_path_selection,
                    use_local_global_attn=use_local_global_attn,
                    use_global_invariance=use_global_invariance,
                    attn_num_heads=attn_num_heads,
                    use_flash_attention=use_flash_attention,
                    use_adaptive_scales=use_adaptive_scales
                )
            elif block_type == "dual_path_conv":
                # Architecture-overhaul variant: shared dilated-Conv core
                block = DualPathConvBlock(
                    d_model=d_model,
                    layer_idx=i,
                    use_path_selection=use_path_selection,
                    use_local_global_attn=use_local_global_attn,
                    use_global_invariance=use_global_invariance,
                    attn_num_heads=attn_num_heads,
                    use_flash_attention=use_flash_attention,
                    use_adaptive_scales=use_adaptive_scales
                )
            elif block_type == "mixed":
                if i % 2 == 0:
                    block = EnhancedMambaBlock(
                        d_model=d_model,
                        use_local_global_attn=use_local_global_attn,
                        use_global_invariance=use_global_invariance,
                        attn_num_heads=attn_num_heads,
                        use_flash_attention=use_flash_attention,
                        use_adaptive_scales=use_adaptive_scales
                    )
                else:
                    block = DualPathMambaBlock(
                        d_model=d_model,
                        layer_idx=i,
                        use_path_selection=use_path_selection,
                        use_local_global_attn=use_local_global_attn,
                        use_global_invariance=use_global_invariance,
                        attn_num_heads=attn_num_heads,
                        use_flash_attention=use_flash_attention,
                        use_adaptive_scales=use_adaptive_scales
                    )
            elif block_type == "enhanced":
                block = EnhancedMambaBlock(
                    d_model=d_model,
                    use_local_global_attn=use_local_global_attn,
                    use_global_invariance=use_global_invariance,
                    attn_num_heads=attn_num_heads,
                    use_flash_attention=use_flash_attention,
                    use_adaptive_scales=use_adaptive_scales
                )
            elif block_type == "basic_transformer":
                # Vanilla-stack core-swap control (control specification)
                block = BasicTransformerBlock(d_model=d_model, nhead=attn_num_heads)
            elif block_type == "basic_conv":
                # Vanilla-stack core-swap control (control specification)
                block = BasicConvBlock(d_model=d_model)
            else:  # "basic"
                block = BasicMambaBlock(d_model=d_model)
            
            self.mamba_blocks.append(block)

        # Whole independent stack on the shared input. Empty unless a
        # two-stack configuration is selected: no extra parameters otherwise.
        self.secondary_blocks = nn.ModuleList()
        if self.has_independent_stack:
            secondary_class = ReadoutConvBlock if block_type == 'dual_mamba_conv' else ReadoutMambaBlock
            self.secondary_blocks.extend(
                secondary_class(d_model=d_model, layer_idx=i) for i in range(n_layer)
            )

        # 4. Sequence pooling layer
        if block_type == "dual_readout_mamba" or self.has_independent_stack:
            self.sequence_pooler = DualReadoutPool(
                d_model, hidden_dim=readout_hidden_dim,
                direct_pool=readout_pool, direct_source=readout_source,
                fusion_mode=readout_fusion, fusion_rank=readout_rank,
            )
        else:
            self.sequence_pooler = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(start_dim=1)
            )
        
        # 5. Output normalization
        self.output_norm = nn.LayerNorm(d_model)
        
        # 6. Classification head ("mlp" = legacy ours; "linear" = baseline-matched)
        if classifier_head == "linear":
            self.classifier = nn.Sequential(
                nn.Linear(d_model, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model // 2, num_classes)
            )
        
        # 7. Contrastive learning projection head
        self.contrastive_projector = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Initialize weights (only initialize non-Transformer parts)
        self._init_weights()
        
        # Move to device
        self.to(self.device)
        
        # Optional: Compile model for optimization
        if compile_model and hasattr(torch, 'compile'):
            try:
                self.forward_sequence = torch.compile(self.forward_sequence)
                print("✅ Model compilation optimization enabled")
            except Exception as e:
                print(f"⚠️ Model compilation failed: {e}")
    
    def _init_weights(self):
        """Weight initialization (only initialize non-Transformer parts)"""
        for name, module in self.named_modules():
            if 'nucleotide_embedding' in name:
                continue
                
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
    
    def forward_sequence(
        self, 
        sequences: List[str],  # Input token sequence list
        variant_type: Optional[str] = None,
        training_mode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for a single sequence type"""
        
        x = self.nucleotide_embedding(sequences, training_mode=training_mode)
        
        # 2. Project to model dimension
        x = self.embedding_projection(x)
        readout_input = x  # Same shared X feeds both v3 network-level paths.
        
        batch_size, seq_len, _ = x.shape
        
        # 3. Pass through Mamba blocks (without residual connections)
        path_weights_list = []
        path_selection_loss = torch.tensor(0.0, device=self.device)
        
        for block in self.mamba_blocks:
            if isinstance(block, (DualPathMambaBlock, DualPathTransformerBlock,
                                  DualPathConvBlock, ParallelMambaConvNeXtBlock,
                                  ReadoutMambaBlock)):
                x, path_weights, block_loss = block(
                    x, 
                    variant_type=variant_type,
                    training_mode=training_mode
                )
                if path_weights is not None:
                    path_weights_list.append(path_weights)
                
                if training_mode and block_loss is not None:
                    path_selection_loss = path_selection_loss + block_loss
            else:
                x = block(x)
        
        # 4. Sequence pooling
        if self.has_independent_stack:
            secondary = self._run_secondary_stack(readout_input, variant_type, training_mode)
            # Reuse the base mean/LN/concat/Linear readout. The pooler's
            # historical 'direct' argument now contains secondary-stack features.
            pooled = self.sequence_pooler(secondary, x)
        elif self.block_type == "dual_readout_mamba":
            # The frozen encoder contract supplies a dense fixed-length tensor.
            # Do not infer a token mask from string lengths. If integration
            # pads variable-length features, supply its true mask to the pooler
            # and apply the same mask policy to the baseline.
            pooled = self.sequence_pooler(readout_input, x)
        else:
            x_pool = x.transpose(1, 2)
            pooled = self.sequence_pooler(x_pool)
        
        # 5. Output normalization
        pooled = self.output_norm(pooled)
        
        # 6. Calculate path selection weights
        if self.use_path_selection and len(path_weights_list) > 0:
            avg_path_weights = torch.stack(path_weights_list).mean(dim=0)
        else:
            avg_path_weights = None
        
        # 7. Classification prediction
        class_pred = self.classifier(pooled)
        
        # 8. Contrastive learning features
        contrastive_feat = self.contrastive_projector(pooled)
        
        # 9. Calculate total path selection loss
        if training_mode and path_selection_loss is not None:
            path_selection_loss = path_selection_loss / len(self.mamba_blocks)
        
        return class_pred, pooled, contrastive_feat, path_selection_loss

    def _run_secondary_stack(self, inputs: torch.Tensor,
                             variant_type: Optional[str] = None,
                             training_mode: bool = True) -> torch.Tensor:
        """Run the independent stack on dense, all-valid encoder features.

        No string-length mask is guessed. Padded variable-length integration
        needs true lengths before both stacks; flipping padding to the front
        and masking only the final pool is not a valid implementation.
        """
        if not self.has_independent_stack:
            raise ValueError("This model has no independent secondary stack")
        _validate_pool_input(inputs, None)
        hidden = (inputs.flip(1).contiguous()
                  if self.block_type == 'bidirectional_mamba' else inputs)
        for block in self.secondary_blocks:
            hidden, _, _ = block(hidden, variant_type=variant_type, training_mode=training_mode)
        # Pooling is permutation invariant: no output flip is needed before
        # the mean. Token-wise fusion, if added later, would need alignment.
        return hidden

    def load_matching_basic_state_dict(self, state_dict):
        """Copy all common weights from a matching BASIC model, after checks.

        Prefer a fresh, per-seed basic initialization for controlled cold-start
        comparisons. A trained checkpoint is a warm start and requires a
        matched warm-start basic control. New readout weights remain nonzero;
        this loader does not make v3 functionally equal to basic.
        """
        if self.block_type != 'dual_readout_mamba':
            raise ValueError("This helper is only for the v3 candidate")
        expected = self.state_dict()
        readout_keys = {key for key in expected if key.startswith('sequence_pooler.')}
        common_keys = set(expected) - readout_keys
        if set(state_dict) != common_keys:
            missing = sorted(common_keys - set(state_dict))
            unexpected = sorted(set(state_dict) - common_keys)
            raise ValueError(f"Not a matching basic state: missing={missing}, unexpected={unexpected}")
        mismatched = [key for key in common_keys if state_dict[key].shape != expected[key].shape]
        if mismatched:
            raise ValueError(f"Basic state shape mismatch: {sorted(mismatched)}")
        return self.load_state_dict(state_dict, strict=False)

    def load_matching_c1_state_dict(self, state_dict):
        """Copy common base-network weights; only new branch/refinement keys may be absent.

        Use a fresh shared initialization for cold-start comparisons.
        Loading a trained base network is a warm start, requiring a matched continuation
        control. A new secondary stack or nonzero refinement changes the
        function; common-weight transfer does not recover the original predictions.
        All key/shape checks precede any mutation. Config provenance must be
        supplied separately: a state_dict cannot prove its input source.
        """
        if ((self.block_type != 'dual_readout_mamba' and not self.has_independent_stack) or self.readout_pool != 'mean'
                or self.readout_source != 'input'):
            raise ValueError("Transfer requires mean readout from the shared input")
        expected = self.state_dict()
        extra = {key for key in expected if key.startswith(('sequence_pooler.refinement.', 'secondary_blocks.'))}
        common = set(expected) - extra
        if set(state_dict) != common:
            missing = sorted(common - set(state_dict))
            unexpected = sorted(set(state_dict) - common)
            raise ValueError(f"Not a matching base-network state: missing={missing}, unexpected={unexpected}")
        mismatched = [key for key in common if state_dict[key].shape != expected[key].shape]
        if mismatched:
            raise ValueError(f"Base-network state shape mismatch: {sorted(mismatched)}")
        return self.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def diagnose_readout(self, sequences: List[str]) -> dict:
        """Eval-only diagnostics on one batch; no persistent activation hooks.

        Zeroing one normalized summary is a dependence probe on THIS model,
        not a separately trained single-path performance result. Recompute
        the refinement after each zeroing; do not leave interactions with a
        removed branch active. Predictions reuse features without dropout.
        """
        if self.block_type != 'dual_readout_mamba' and not self.has_independent_stack:
            raise ValueError("diagnose_readout requires a network-level dual readout")
        if any(module.training for module in self.modules()):
            raise ValueError("Call model.eval() before readout diagnostics")
        inputs = self.embedding_projection(self.nucleotide_embedding(sequences, training_mode=False))
        hidden = inputs
        for block in self.mamba_blocks:
            hidden, _, _ = block(hidden, variant_type=None, training_mode=False)
        pool = self.sequence_pooler
        secondary = (self._run_secondary_stack(inputs, training_mode=False)
                     if self.has_independent_stack else inputs)
        main, direct, weights = pool.summaries(secondary, hidden)
        # Linear contributions BEFORE output LayerNorm. The nonlinear term
        # cannot in general be assigned uniquely to either path.
        left, right = pool.fusion.weight.split(self.d_model, dim=1)
        main_part = F.linear(main, left)
        direct_part = F.linear(direct, right)
        linear = pool.fuse(main, direct, include_refinement=False)
        refinement = (torch.zeros_like(linear) if pool.refinement is None else
                      pool.refinement(main, direct))
        fused = pool.fuse(main, direct)
        predictions = {
            'full': self.classifier(self.output_norm(fused)),
            'without_direct': self.classifier(self.output_norm(pool.fuse(main, torch.zeros_like(direct)))),
            'without_main': self.classifier(self.output_norm(pool.fuse(torch.zeros_like(main), direct))),
            'without_refinement': self.classifier(self.output_norm(linear)),
        }
        weights = weights.float()
        entropy = -(weights * weights.clamp_min(1e-30).log()).sum(dim=1)
        length = weights.shape[1]
        normalized_entropy = entropy / max(float(torch.tensor(float(length)).log()), 1e-12)
        main_rms = main_part.float().square().mean().sqrt()
        direct_rms = direct_part.float().square().mean().sqrt()
        metrics = {
            'main_projected_rms': main_rms.item(),
            'direct_projected_rms': direct_rms.item(),
            'direct_to_main_rms_ratio': (direct_rms / main_rms.clamp_min(1e-12)).item(),
            'projected_cosine_mean': F.cosine_similarity(main_part.float(), direct_part.float(), dim=-1).mean().item(),
            'attention_entropy_normalized': normalized_entropy.mean().item(),
            'attention_effective_tokens': weights.square().sum(dim=1).reciprocal().mean().item(),
            'attention_max_weight': weights.max(dim=1).values.mean().item(),
            'remove_direct_logit_rms': (predictions['full'].float() - predictions['without_direct'].float()).square().mean().sqrt().item(),
            'remove_main_logit_rms': (predictions['full'].float() - predictions['without_main'].float()).square().mean().sqrt().item(),
            'refinement_rms': refinement.float().square().mean().sqrt().item(),
            'refinement_to_linear_rms_ratio': (refinement.float().square().mean().sqrt() /
                                              linear.float().square().mean().sqrt().clamp_min(1e-12)).item(),
            'remove_refinement_logit_rms': (predictions['full'].float() - predictions['without_refinement'].float()).square().mean().sqrt().item(),
        }
        if self.has_independent_stack:
            # Explicit names; retain old direct-path keys for log readers.
            predictions['without_secondary'] = predictions['without_direct']
            metrics['secondary_projected_rms'] = metrics['direct_projected_rms']
            metrics['secondary_to_main_rms_ratio'] = metrics['direct_to_main_rms_ratio']
            metrics['remove_secondary_logit_rms'] = metrics['remove_direct_logit_rms']
        return {'metrics': metrics, 'logits': {key: value.detach().cpu() for key, value in predictions.items()}}
    
    def forward(
        self,
        # Input data: sequence lists for different variant types
        original_sequences: List[str],
        labels: torch.Tensor,
        semantic_sequences: Optional[List[str]] = None,
        confusion_sequences: Optional[List[str]] = None,
        random_mutation_sequences: Optional[List[str]] = None,
        training_mode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            original_sequences: List of original token sequences
            labels: Label tensor
            semantic_sequences: List of semantic variant sequences
            confusion_sequences: List of confusion variant sequences
            random_mutation_sequences: List of random mutation variant sequences
            training_mode: Whether in training mode
            
        Returns:
            orig_class_pred: Original sequence classification prediction
            semantic_class_pred: Semantic variant classification prediction
            confusion_class_pred: Confusion variant classification prediction
            total_aux_loss: Total auxiliary loss
            orig_contrastive_feat: Original sequence contrastive features
            all_contrastive_feat: Contrastive features for all sequences
        """
        batch_size = len(original_sequences)
        
        # Ensure labels are on correct device
        labels = labels.to(self.device)
        
        # Process original sequences
        orig_class_pred, orig_pooled_feat, orig_contrastive_feat, orig_path_loss = self.forward_sequence(
            original_sequences, 
            variant_type="original" if training_mode else None,
            training_mode=training_mode
        )
        
        # Process semantic variants
        semantic_class_pred = None
        semantic_contrastive_feat = None
        semantic_path_loss = torch.tensor(0.0, device=self.device)
        
        if semantic_sequences is not None and semantic_sequences:
            semantic_class_pred, _, semantic_contrastive_feat, semantic_path_loss = self.forward_sequence(
                semantic_sequences, 
                variant_type="semantic" if training_mode else None,
                training_mode=training_mode
            )
        
        # Create default values
        if semantic_class_pred is None:
            semantic_class_pred = torch.zeros(batch_size, self.num_classes, device=self.device)
            semantic_contrastive_feat = torch.zeros(batch_size, self.projection_dim, device=self.device)
        
        # Process confusion variants
        confusion_class_pred = None
        confusion_contrastive_feat = None
        confusion_path_loss = torch.tensor(0.0, device=self.device)
        
        if confusion_sequences is not None and confusion_sequences:
            confusion_class_pred, _, confusion_contrastive_feat, confusion_path_loss = self.forward_sequence(
                confusion_sequences, 
                variant_type="confusion" if training_mode else None,
                training_mode=training_mode
            )
        
        if confusion_class_pred is None:
            confusion_class_pred = torch.zeros(batch_size, self.num_classes, device=self.device)
            confusion_contrastive_feat = torch.zeros(batch_size, self.projection_dim, device=self.device)
        
        # Process random mutation variants
        random_mutation_contrastive_feat = None
        random_mutation_path_loss = torch.tensor(0.0, device=self.device)
        
        if random_mutation_sequences is not None and random_mutation_sequences:
            _, _, random_mutation_contrastive_feat, random_mutation_path_loss = self.forward_sequence(
                random_mutation_sequences,
                variant_type=None,
                training_mode=training_mode
            )
        
        if random_mutation_contrastive_feat is None:
            random_mutation_contrastive_feat = torch.zeros(batch_size, self.projection_dim, device=self.device)
        
        # Calculate total path selection loss
        total_path_selection_loss = orig_path_loss + semantic_path_loss + confusion_path_loss
        
        # Total auxiliary loss
        total_aux_loss = total_path_selection_loss * self.path_selection_weight
        
        # Merge contrastive features
        contrastive_features_list = [
            orig_contrastive_feat,
            semantic_contrastive_feat,
            confusion_contrastive_feat,
            random_mutation_contrastive_feat
        ]
        
        all_contrastive_feat = torch.cat(contrastive_features_list, dim=0)
        
        return (
            orig_class_pred,
            semantic_class_pred,
            confusion_class_pred,
            total_aux_loss,
            orig_contrastive_feat,
            all_contrastive_feat
        )
    
    def clear_cache(self):
        """Clear feature cache"""
        self.nucleotide_embedding.clear_cache()
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return self.nucleotide_embedding.get_cache_stats()
    
    def get_model_summary(self) -> dict:
        """Get model statistics"""
        if hasattr(self.nucleotide_embedding, 'get_model_info'):
            encoder_info = self.nucleotide_embedding.get_model_info()
        else:
            # The shape-only stand-in intentionally has no metadata API.
            encoder_info = {
                'total_params': sum(p.numel() for p in self.nucleotide_embedding.parameters()),
                'trainable_params': sum(p.numel() for p in self.nucleotide_embedding.parameters()
                                        if p.requires_grad),
            }
        encoder_params = encoder_info['total_params']
        encoder_trainable = encoder_info['trainable_params']
        
        # Keep the historical mamba_parameters alias, but count ALL downstream
        # modules, including secondary stacks; exclude the encoder explicitly.
        encoder_parameter_ids = {id(p) for p in self.nucleotide_embedding.parameters()}
        downstream = [p for p in self.parameters() if id(p) not in encoder_parameter_ids]
        mamba_params = sum(p.numel() for p in downstream)
        downstream_trainable = sum(p.numel() for p in downstream if p.requires_grad)
        
        total_params = encoder_params + mamba_params
        trainable_params = encoder_trainable + downstream_trainable
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_parameters': encoder_params,
            'encoder_trainable': encoder_trainable,
            'mamba_parameters': mamba_params,
            'downstream_parameters': mamba_params,
            'downstream_trainable_parameters': downstream_trainable,
            'main_stack_parameters': sum(p.numel() for p in self.mamba_blocks.parameters()),
            'secondary_stack_parameters': sum(p.numel() for p in self.secondary_blocks.parameters()),
            'secondary_stack_type': ('conv' if self.block_type == 'dual_mamba_conv' else
                                     'reverse_mamba' if self.block_type == 'bidirectional_mamba' else None),
            'embedding_dim': self.embedding_dim,
            'd_model': self.d_model,
            'n_layer': self.n_layer,
            'block_type': self.block_type,
            'use_path_selection': self.use_path_selection,
            'use_local_global_attn': self.use_local_global_attn,
            'use_global_invariance': self.use_global_invariance,
            'use_adaptive_scales': self.use_adaptive_scales,
            'parameter_mb': total_params * 4 / (1024 * 1024),
            'model_architecture': (
                'Encoder + independent basic Mamba / dilated Conv stacks + linear readout'
                if self.block_type == 'dual_mamba_conv' else
                'Encoder + independent forward / backward basic Mamba stacks + linear readout'
                if self.block_type == 'bidirectional_mamba' else
                'Encoder + parallel basic Mamba mean readout / direct feature readout'
                if self.block_type == 'dual_readout_mamba' else
                'Encoder + parallel Mamba / ConvNeXt-style 1D correction'
                if self.block_type == 'parallel_mamba_convnext' else
                'NTv3_8M_pre (fully trainable) + Mamba (optimized simplified enhancement)'
            ),
            'auxiliary_branch_parameters': sum(
                sum(p.numel() for p in block.aux_branch.parameters()) + block.alpha.numel()
                for block in self.mamba_blocks
                if isinstance(block, ParallelMambaConvNeXtBlock)
            ),
            'readout_parameters': sum(p.numel() for p in self.sequence_pooler.parameters()),
            'readout_pool': self.readout_pool if self.block_type == 'dual_readout_mamba' or self.has_independent_stack else None,
            'readout_source': ('secondary_stack' if self.has_independent_stack else
                               self.readout_source if self.block_type == 'dual_readout_mamba' else None),
            'readout_fusion': self.readout_fusion if self.block_type == 'dual_readout_mamba' or self.has_independent_stack else None,
            'readout_rank': self.readout_rank if self.block_type == 'dual_readout_mamba' else None,
            'fusion_refinement_parameters': (
                sum(p.numel() for p in self.sequence_pooler.refinement.parameters())
                if self.block_type == 'dual_readout_mamba' and self.sequence_pooler.refinement is not None else 0
            ),
        }
    
    def print_detailed_summary(self):
        """Print detailed model summary"""
        summary = self.get_model_summary()
        
        print(f"\n{'='*60}")
        print(f"🧬 NTv3 + Mamba Model Detailed Summary (Optimized Simplified Enhancement)")
        print(f"{'='*60}")
        print(f"📊 Parameter Statistics:")
        print(f"  Total parameters: {summary['total_parameters']:,}")
        print(f"  Trainable parameters: {summary['trainable_parameters']:,}")
        print(f"  NTv3 parameters: {summary['encoder_parameters']:,}")
        print(f"  NTv3 trainable parameters: {summary['encoder_trainable']:,}")
        print(f"  Mamba part parameters: {summary['mamba_parameters']:,}")
        print(f"  Total memory usage: {summary['parameter_mb']:.2f} MB")
        
        print(f"\n📈 Architecture Configuration:")
        print(f"  Model architecture: {summary['model_architecture']}")
        print(f"  Embedding dimension: {summary['embedding_dim']}")
        print(f"  Model dimension: {summary['d_model']}")
        print(f"  Mamba layers: {summary['n_layer']}")
        print(f"  Block type: {summary['block_type']}")
        print(f"  Path selection: {summary['use_path_selection']}")
        if self.has_independent_stack:
            print("  Main path: unchanged basic Mamba stack -> mean -> LN")
            print(f"  Secondary path: independent {summary['secondary_stack_type']} stack -> mean -> LN")
            print("  Fusion: base Linear(2d,d), without refinement")
            print(f"  Secondary-stack parameters: {summary['secondary_stack_parameters']:,}")
            print(f"  Readout parameters: {summary['readout_parameters']:,}")
        elif self.block_type == 'dual_readout_mamba':
            print("  Main path: unchanged basic Mamba stack -> mean -> LN")
            print(f"  Direct path: {self.readout_source} -> {self.readout_pool} pool -> LN")
            if self.readout_fusion == 'linear':
                print("  Fusion: base Linear(2d,d), without refinement")
            else:
                print(f"  Fusion: base Linear(2d,d) + {self.readout_fusion} refinement; rank={self.readout_rank}")
            print(f"  Readout parameters: {summary['readout_parameters']:,}")
            print(f"  New refinement parameters: {summary['fusion_refinement_parameters']:,}")
        elif self.block_type == 'parallel_mamba_convnext':
            print("  Main path: unchanged BasicMambaBlock")
            print("  Auxiliary path: pre-LN, DWConv1d(k=7), channel MLP (2x)")
            print("  Fusion: main(X) + alpha * auxiliary(X); alpha starts at zero")
            print(f"  Auxiliary parameters (including alpha): {summary['auxiliary_branch_parameters']:,}")
        else:
            print(f"  Functional semantics path: {summary['use_local_global_attn']} (simplified global attention)")
            print(f"  Attack response path: {summary['use_global_invariance']} (multi-scale cross-segment attention)")
            print(f"  Adaptive scales: {summary['use_adaptive_scales']}")
        
        print(f"\n🧬 NTv3 Configuration:")
        print(f"  Model repository: {self.transformer_model_repo}")
        print(f"  Maximum sequence length: {self.max_seq_len}")
        print(f"  Use caching: {self.use_caching}")
        print(f"  Trust remote code: {self.trust_remote_code}")
        
        if hasattr(self.nucleotide_embedding, 'get_model_info'):
            encoder_info = self.nucleotide_embedding.get_model_info()
            print(f"  NTv3 architecture: {encoder_info.get('architecture', 'U-Net conv tower → Transformer stack → deconv tower → LM head')}")
            print(f"  NTv3 vocabulary size: {encoder_info.get('vocab_size', 'N/A')}")
            print(f"  NTv3 hidden size: {encoder_info.get('hidden_size', 'N/A')}")
        
        print(f"\n🔧 Enhancement Module Optimization Features:")
        if self.has_independent_stack:
            print("  Independent branch weights; no scalar alpha or per-layer coupling")
            print("  Encoder called once per view; heads remain parallel; path loss is zero")
            print("  Dense fixed-length features only; validation does not imply an AUC gain")
        elif self.block_type == 'dual_readout_mamba':
            print("  Network-level dual path; main blocks are unchanged basic computations")
            print("  No alpha, route supervision, internal branch dropout or quadratic attention")
            print("  Nonzero gradient access at initialization is NOT an AUC guarantee")
        elif self.block_type == 'parallel_mamba_convnext':
            print("  Two parallel input branches; one Mamba call per block")
            print("  No route network, path KL/entropy, concatenation fusion or extra post-norm")
            print("  alpha=0 recovers the matching basic block in deterministic evaluation")
        else:
            print(f"  1. Efficient segment splitting: Using unfold and precomputation optimization")
            print(f"  2. Dimensionality reduction: Reduce dimension before attention, reducing computation")
            print(f"  3. Flash Attention: Support for efficient attention computation")
            print(f"  4. Adaptive scales: Dynamic adjustment based on sequence length")
            print(f"  5. Shared Mamba: Both paths share parameters; two core evaluations per block")
            print(f"  6. Memory optimization: Avoid unnecessary reshaping and padding")
        
        print(f"{'='*60}")


def create_nucleotide_mamba_model(
    transformer_model_repo: str = "InstaDeepAI/NTv3_8M_pre",
    embedding_dim: int = 256,
    # Mamba model configuration
    d_model: int = 256,
    n_layer: int = 4,
    block_type: str = "dual_path",
    use_path_selection: bool = True,
    # Enhancement module parameters
    use_local_global_attn: bool = True,
    use_global_invariance: bool = True,
    use_flash_attention: bool = True,
    use_adaptive_scales: bool = True,  # New parameter
    # Transformer configuration
    freeze_transformer: bool = True,  # Fully trainable
    use_caching: bool = True,
    trust_remote_code: bool = True,
    # GPU optimization parameters
    enable_tf32: bool = True,
    compile_model: bool = False,
    **kwargs
) -> NucleotideMambaModel:
    """Convenience function to create NTv3 + Mamba model"""

    if isinstance(block_type, str) and block_type.startswith("v6:"):
        from . import downstream_fusion as _v6
        _no_grn = isinstance(block_type, str) and block_type.endswith("+nogrn")
        if _no_grn:
            block_type = block_type[:-len("+nogrn")]
        _dev = kwargs.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        _enc = _RealEncoder(
            model_repo=transformer_model_repo,
            output_dim=embedding_dim,
            max_seq_len=kwargs.get("max_seq_len", 2048),
            use_cache=False,
            device=str(_dev),
            trust_remote_code=trust_remote_code,
            freeze_transformer=freeze_transformer,
        )
        _parts = block_type.split(":")
        _extra = {}
        if len(_parts) > 2:
            _extra["main_kernel"] = _parts[2]
        if _no_grn:
            _extra["no_grn"] = True
        return _v6.create_v6_network(_parts[1], encoder=_enc, device=_dev,
                                     transformer_model_repo=transformer_model_repo, **_extra)
    default_kwargs = {
        'transformer_model_repo': transformer_model_repo,
        'embedding_dim': embedding_dim,
        'd_model': d_model,
        'n_layer': n_layer,
        'projection_dim': 128,
        'num_classes': 1,
        'variant_specialization_weight': 0.2,
        'block_type': block_type,
        'dropout_rate': 0.1,
        'use_path_selection': use_path_selection,
        'path_selection_weight': 0.1,
        'max_seq_len': 512,
        # Enhancement module configuration
        'use_local_global_attn': use_local_global_attn,
        'use_global_invariance': use_global_invariance,
        'attn_num_heads': 4,
        'use_flash_attention': use_flash_attention,
        'use_adaptive_scales': use_adaptive_scales,
        # Transformer configuration
        'freeze_transformer': freeze_transformer,
        'use_caching': use_caching,
        'trust_remote_code': trust_remote_code,
        # GPU optimization
        'enable_tf32': enable_tf32,
        'compile_model': compile_model
    }
    
    # Update default parameters
    default_kwargs.update(kwargs)
    
    return NucleotideMambaModel(**default_kwargs)



# The default is the locked configuration.
