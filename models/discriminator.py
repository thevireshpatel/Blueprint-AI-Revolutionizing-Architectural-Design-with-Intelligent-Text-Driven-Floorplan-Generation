import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, annotation_dim=0):
        super(Discriminator, self).__init__()
        
        self.annotation_dim = annotation_dim
        
        if annotation_dim > 0:
            self.annotation_fc = nn.Linear(annotation_dim, 256)
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, annotations=None):
        img_features = self.model[:-1](img)
        
        if annotations is not None:
            annotation_features = self.annotation_fc(annotations)
            annotation_features = annotation_features.unsqueeze(-1).unsqueeze(-1)
            combined_features = img_features + annotation_features
            return self.model[-1](combined_features)
        
        return self.model[-1](img_features)