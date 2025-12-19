import torch.nn as nn
import timm

class EfficientNetBinary(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_b0', pretrained=pretrained, num_classes=0)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze 2 block cuối
        for block in self.backbone.blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True
        
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(1)