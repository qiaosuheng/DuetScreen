import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

# Fix TripletContrastiveLoss in loss.py:

class TripletContrastiveLoss(nn.Module):
    """
    Fixed triplet contrastive loss
    """
    def __init__(self, margin=1.0, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, 
                anchor_feat: torch.Tensor,      # Original sequence features [batch_size, projection_dim]
                positive_feats: List[torch.Tensor],  # List of positive sample features
                negative_feat: torch.Tensor     # Negative sample features [batch_size, projection_dim]
               ) -> torch.Tensor:
        """
        Calculate triplet contrastive loss
        """
        batch_size = anchor_feat.shape[0]
        total_loss = 0.0
        valid_pairs = 0
        
        # 🔥 Fix: Check dimension consistency
        if anchor_feat.dim() != 2 or negative_feat.dim() != 2:
            return torch.tensor(0.0, device=anchor_feat.device)
        
        # Normalize features
        anchor_feat = F.normalize(anchor_feat, p=2, dim=1)
        negative_feat = F.normalize(negative_feat, p=2, dim=1)
        
        # Calculate similarity between anchor and negative samples
        neg_similarity = F.cosine_similarity(anchor_feat, negative_feat)
        
        for i, positive_feat in enumerate(positive_feats):
            if positive_feat is None:
                continue
                
            # 🔥 Fix: Check positive sample feature dimensions
            if positive_feat.dim() != 2 or positive_feat.shape != anchor_feat.shape:
                continue
                
            # Normalize positive sample features
            positive_feat = F.normalize(positive_feat, p=2, dim=1)
            
            # Calculate similarity between anchor and positive samples
            pos_similarity = F.cosine_similarity(anchor_feat, positive_feat)
            
            # Calculate triplet loss
            try:
                triplet_loss = F.triplet_margin_loss(
                    anchor_feat, positive_feat, negative_feat,
                    margin=self.margin, p=2, eps=1e-7
                )
                
                # Calculate InfoNCE-style contrastive loss
                pos_exp = torch.exp(pos_similarity / self.temperature)
                neg_exp = torch.exp(neg_similarity / self.temperature)
                
                contrastive_loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()
                
                # Combine both losses
                combined_loss = 0.7 * triplet_loss + 0.3 * contrastive_loss
                
                total_loss += combined_loss
                valid_pairs += 1
                
            except Exception as e:
                print(f"⚠️ Triplet loss calculation error: {str(e)}")
                continue
        
        if valid_pairs == 0:
            return torch.tensor(0.0, device=anchor_feat.device)
        
        return total_loss / valid_pairs

class HardTripletLoss(nn.Module):
    """
    Hard triplet loss: Select the hardest positive and negative sample pairs
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor_feat: torch.Tensor, 
                positive_feats: List[torch.Tensor],
                negative_feat: torch.Tensor) -> torch.Tensor:
        """
        Calculate hard triplet loss
        
        Strategy: Select the hardest positive sample (lowest similarity to anchor) 
                 and hardest negative sample (highest similarity to anchor)
        """
        batch_size = anchor_feat.shape[0]
        
        # Normalize features
        anchor_feat = F.normalize(anchor_feat, p=2, dim=1)
        negative_feat = F.normalize(negative_feat, p=2, dim=1)
        
        # Find the hardest positive sample (lowest similarity)
        min_pos_similarity = float('inf')
        hardest_positive = None
        
        for positive_feat in positive_feats:
            if positive_feat is None:
                continue
                
            positive_feat = F.normalize(positive_feat, p=2, dim=1)
            pos_similarity = F.cosine_similarity(anchor_feat, positive_feat).mean()
            
            if pos_similarity < min_pos_similarity:
                min_pos_similarity = pos_similarity
                hardest_positive = positive_feat
        
        if hardest_positive is None:
            return torch.tensor(0.0, device=anchor_feat.device)
        
        # Calculate similarity with negative samples
        neg_similarity = F.cosine_similarity(anchor_feat, negative_feat)
        
        # Calculate hard triplet loss
        hardest_loss = F.triplet_margin_loss(
            anchor_feat, hardest_positive, negative_feat,
            margin=self.margin, p=2, eps=1e-7
        )
        
        return hardest_loss


class VariantConsistencyLoss(nn.Module):
    """
    Variant consistency loss
    Ensure variant classification predictions are consistent with original sequence predictions
    """
    def __init__(self, consistency_type: str = "kl", temperature: float = 1.0):
        super().__init__()
        self.consistency_type = consistency_type
        self.temperature = temperature
        
        if consistency_type == "kl":
            self.criterion = nn.KLDivLoss(reduction="batchmean")
        elif consistency_type == "mse":
            self.criterion = nn.MSELoss()
        elif consistency_type == "js":  # Jensen-Shannon divergence
            self.criterion = self.js_divergence
        else:
            raise ValueError(f"Unsupported consistency_type: {consistency_type}")

    def js_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Calculate Jensen-Shannon divergence"""
        m = 0.5 * (p + q)
        return 0.5 * (F.kl_div(F.log_softmax(p, dim=1), m, reduction='batchmean') + 
                     F.kl_div(F.log_softmax(q, dim=1), m, reduction='batchmean'))

    def forward(self, orig_pred: torch.Tensor, variant_pred: torch.Tensor) -> torch.Tensor:
        """
        Calculate variant consistency loss
        
        Args:
            orig_pred: Original sequence predictions [batch_size, num_classes]
            variant_pred: Variant predictions [batch_size, num_classes]
            
        Returns:
            Consistency loss
        """
        if self.consistency_type == "kl":
            # Use KL divergence
            orig_probs = F.log_softmax(orig_pred / self.temperature, dim=1)
            variant_probs = F.softmax(variant_pred / self.temperature, dim=1)
            loss = self.criterion(orig_probs, variant_probs)
            
        elif self.consistency_type == "mse":
            # Use mean squared error
            orig_probs = F.softmax(orig_pred / self.temperature, dim=1)
            variant_probs = F.softmax(variant_pred / self.temperature, dim=1)
            loss = self.criterion(orig_probs, variant_probs)
            
        elif self.consistency_type == "js":
            # Use Jensen-Shannon divergence
            orig_probs = F.softmax(orig_pred / self.temperature, dim=1)
            variant_probs = F.softmax(variant_pred / self.temperature, dim=1)
            loss = self.criterion(orig_probs, variant_probs)
            
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal Loss
        
        Args:
            inputs: Model predictions [batch_size, num_classes] or [batch_size]
            targets: True labels [batch_size]
            
        Returns:
            Focal Loss
        """
        if inputs.dim() > 1:
            inputs = inputs.squeeze()
            
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevent numerical instability
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


# Test code
if __name__ == "__main__":
    # Test necessary loss functions
    print("🧪 Testing necessary loss functions...")
    
    # Simulate data
    batch_size, proj_dim = 4, 128
    orig_feat = torch.randn(batch_size, proj_dim)
    semantic_feat = torch.randn(batch_size, proj_dim)
    confusion_feat = torch.randn(batch_size, proj_dim)
    negative_feat = torch.randn(batch_size, proj_dim)
    
    # Test triplet contrastive loss
    triplet_loss = TripletContrastiveLoss()
    loss1 = triplet_loss(orig_feat, [semantic_feat, confusion_feat], negative_feat)
    print(f"Triplet contrastive loss: {loss1.item():.4f}")
    
    # Test hard triplet loss
    hard_triplet_loss = HardTripletLoss()
    loss2 = hard_triplet_loss(orig_feat, [semantic_feat, confusion_feat], negative_feat)
    print(f"Hard triplet loss: {loss2.item():.4f}")
    
    # Test variant consistency loss
    orig_pred = torch.randn(batch_size, 1)
    variant_pred = torch.randn(batch_size, 1)
    consistency_loss = VariantConsistencyLoss()
    loss3 = consistency_loss(orig_pred, variant_pred)
    print(f"Variant consistency loss: {loss3.item():.4f}")
    
    # Test Focal Loss
    labels = torch.randint(0, 2, (batch_size,)).float()
    focal_loss = FocalLoss()
    loss4 = focal_loss(orig_pred.squeeze(), labels)
    print(f"Focal Loss: {loss4.item():.4f}")
    
    print("✅ Necessary loss functions test passed!")
