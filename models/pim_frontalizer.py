import torch
import torch.nn as nn
import torch.nn.functional as F

class D2SCGANSuperRes(nn.Module):
    """
    Dual Deep-Shallow Channeled GAN (D2SC-GAN) for Low-Resolution Faces in the Wild.
    Dual-path architecture:
    - Shallow Channel: Preserves high-frequency edge gradients and structural landmarks.
    - Deep Channel: Synthesizes high-level semantic identity features.
    """
    def __init__(self):
        super(D2SCGANSuperRes, self).__init__()
        
        # Shallow path (High-frequency edge preservation)
        self.shallow_path = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        
        # Deep path (Semantic reconstruction)
        self.deep_path = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.PReLU()
        )
        
        # Fusion head
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_shallow = self.shallow_path(x)
        feat_deep = self.deep_path(x)
        concat_feat = torch.cat([feat_shallow, feat_deep], dim=1)
        sr_img = self.fusion(concat_feat)
        return sr_img

class PIMFrontalizationGAN(nn.Module):
    """
    Pose-Invariant Model (PIM) Frontalization Network (Zhao et al., CVPR 2018).
    Synthesizes a canonical frontal, clean-illuminated face view from an extreme profile/yaw rotated face crop.
    """
    def __init__(self):
        super(PIMFrontalizationGAN, self).__init__()
        
        self.super_res = D2SCGANSuperRes()
        
        # Encoder: compresses profile image into latent pose-free identity code z
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), # 56x56
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 28x28
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 14x14
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Symmetric Decoder: synthesizes frontalized canonical face
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1), # 112x112
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, yaw_angle: float = 0.0) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 3, 112, 112) input face crop [-1, 1]
            yaw_angle: estimated head yaw in degrees
        Returns:
            frontalized_x: (batch_size, 3, 112, 112) canonical frontalized face
        """
        # If pose yaw is mild (< 20 degrees), bypass generative frontalization
        if abs(yaw_angle) < 20.0:
            return x
            
        # First enhance low-resolution details
        sr_x = self.super_res(x)
        
        # Encode profile and decode frontal canonical image
        z = self.encoder(sr_x)
        frontalized_x = self.decoder(z)
        
        # Enforce bilateral symmetry blending
        flipped_x = torch.flip(frontalized_x, dims=[3])
        sym_frontalized = 0.5 * (frontalized_x + flipped_x)
        
        return sym_frontalized
