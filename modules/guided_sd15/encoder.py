import torch
from torch import nn
from torch.nn import functional as F
from .decoder import VAE_AttentionBlock, VAE_ResidualBlock



class VAE_Encoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),              #  B 128 H W
            VAE_ResidualBlock(128, 128),                              #  B 128 H W
            VAE_ResidualBlock(128, 128),                              #  B 128 H W
            
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),  #  B 128 H/2 W/2
            VAE_ResidualBlock(128, 256),                              #  B 256 H/2 W/2
            VAE_ResidualBlock(256, 256),                              #  B 256 H/2 W/2
            
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),  #  B 256 H/4 W/4
            VAE_ResidualBlock(256, 512),                              #  B 512 H/4 W/4
            VAE_ResidualBlock(512, 512),                              #  B 512 H/4 W/4
            
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),  #  B 512 H/8 W/8
            VAE_ResidualBlock(512, 512),                              #  B 512 H/8 W/8
            VAE_ResidualBlock(512, 512),                              #  B 512 H/8 W/8
            VAE_ResidualBlock(512, 512),                              #  B 512 H/8 W/8
            VAE_AttentionBlock(512),                                  #  B 512 H/8 W/8
            VAE_ResidualBlock(512, 512),                              #  B 512 H/8 W/8

            nn.GroupNorm(32, 512),
            nn.SiLU(), 
            
            nn.Conv2d(512, 8, kernel_size=3, padding=1),              #  B 8 H/8 W/8
            nn.Conv2d(8, 8, kernel_size=1, padding=0),                #  B 8 H/8 W/8
        )
    
    def forward(self,
                x: torch.Tensor,    # B C H W
                noise: torch.Tensor  # B C H\8 W\8
                ) -> torch.Tensor:

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))  # left, right, top, bottom

            x = module(x)


        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        
        x = mean + stdev * noise
        x *= 0.18215
        
        return x
        
        
        
        
        
    













