import torch.nn as nn
import timm
import os

class ConvNextBinary(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load ConvNeXt Tiny
        self.backbone = timm.create_model('convnext_tiny', pretrained=pretrained, num_classes=0)
        
        # Freeze toàn bộ
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze 2 stage cuối (Stage 2 và 3)
        for stage in self.backbone.stages[-2:]:
            for param in stage.parameters():
                param.requires_grad = True

        # Head mới
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Flatten(1),
            nn.Dropout(0.5),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(1)