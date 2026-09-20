import torch
import torch.nn as nn
import torch.nn.functional as F

class ANetAttributeParser(nn.Module):
    """
    ANet: Deep Learning Face Attributes in the Wild (Liu et al., ICCV 2015)
    
    Multi-task convolutional attribute predictor operating on 40 facial attributes.
    Generates a spatial occlusion attention mask M_spatial to suppress occluded facial zones
    (e.g., masks, sunglasses, hats).
    """
    ATTRIBUTE_NAMES = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
        'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
        'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]

    def __init__(self, num_attributes: int = 40):
        super(ANetAttributeParser, self).__init__()
        self.num_attributes = num_attributes
        
        # Shared feature extractor backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 56x56
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Attribute classification head (40 multi-task logits)
        self.attribute_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_attributes)
        )
        
        # Spatial Occlusion Mask Generator (1xHxW attention map)
        self.spatial_mask_head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid() # Occlusion weight map [0, 1]
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch_size, 3, 112, 112) aligned face crop patch
        Returns:
            attr_logits: (batch_size, 40) binary attribute prediction logits
            occlusion_mask: (batch_size, 1, 7, 7) spatial occlusion attention map
            is_occluded: (batch_size,) boolean indicator if major occlusion is detected
        """
        feat_map = self.features(x)
        attr_logits = self.attribute_head(feat_map)
        occlusion_mask = self.spatial_mask_head(feat_map)
        
        # Occlusion heuristic check: Eyeglasses (index 15), Wearing_Hat (index 35), or low mean mask weight
        attr_probs = torch.sigmoid(attr_logits)
        has_glasses = attr_probs[:, 15] > 0.5
        has_hat = attr_probs[:, 35] > 0.5
        low_spatial_weight = occlusion_mask.mean(dim=(1, 2, 3)) < 0.60
        
        is_occluded = has_glasses | has_hat | low_spatial_weight
        
        return attr_logits, occlusion_mask, is_occluded
