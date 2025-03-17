import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, annotation_dim=0):
        super(Generator, self).__init__()
        
        input_dim = latent_dim + annotation_dim
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, annotations=None):
        if annotations is not None:
            noise = torch.cat([noise, annotations], dim=1)
        
        noise = noise.unsqueeze(-1).unsqueeze(-1)
        return self.model(noise)