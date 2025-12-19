import torch.nn as nn
import timm

class ViTBinary(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze 2 block cuối và Norm layer
        for block in self.backbone.blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True
        
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
            
        # Thay thế Head
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        # Đảm bảo head luôn được train
        for param in self.backbone.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        output = self.backbone(x)
        return output.squeeze(1)