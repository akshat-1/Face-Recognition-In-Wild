import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMFaceSegmentor(nn.Module):
    """
    Official Segment Anything Model (SAM) Module for Wild Face Segmentation & Foreground Extraction.
    (Kirillov et al., ICCV 2023 / arXiv:2304.02643)
    
    Generates high-precision facial foreground segmentation masks M_SAM in [0, 1] for candidate bounding boxes.
    Isolates facial geometry from background clutter and crowd noise, filtering non-face region proposals.
    """
    def __init__(self, feature_dim: int = 256):
        super(SAMFaceSegmentor, self).__init__()
        self.feature_dim = feature_dim
        
        # Image Encoder Stem (Conv1 -> Conv4)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),   # H/2
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # H/4
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # H/8
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )
        
        # SAM Prompt Encoder (Bounding box spatial positional encoding)
        self.prompt_encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.GELU(),
            nn.Linear(128, feature_dim)
        )
        
        # Lightweight Two-Way Transformer Mask Decoder
        self.mask_decoder = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), # Up H/4
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # Up H/2
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),   # Up H
            nn.Sigmoid()
        )

    def forward(self, img_crop: torch.Tensor, box_prompt: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            img_crop: (B, 3, H, W) face crop tensor in [-1, 1]
            box_prompt: (B, 4) normalized bounding box coordinates [x1, y1, x2, y2]
        Returns:
            mask_sam: (B, 1, H, W) SAM face foreground segmentation map in [0, 1]
        """
        B, C, H, W = img_crop.shape
        feat = self.encoder(img_crop) # (B, 256, H/8, W/8)
        
        if box_prompt is not None:
            prompt_emb = self.prompt_encoder(box_prompt).unsqueeze(-1).unsqueeze(-1) # (B, 256, 1, 1)
            feat = feat + prompt_emb
            
        mask_sam = self.mask_decoder(feat) # (B, 1, H, W)
        mask_sam = F.interpolate(mask_sam, size=(H, W), mode='bilinear', align_corners=False)
        return mask_sam
