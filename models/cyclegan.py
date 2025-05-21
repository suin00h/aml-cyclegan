from torch import nn

from generator import Generator
from discriminator import Discriminator

class CycleGAN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net_G = Generator()        # maps X to Y
        self.net_F = Generator()        # maps Y to X
        self.net_Dx = Discriminator()   # distinguish between images x and translated images F(y)
        self.net_Dy = Discriminator()   # distinguish between images y and translated images G(x)
    
    def forward(self, real_x, real_y):
        fake_y = self.net_G(real_x)     # X -> G(X)
        recon_x = self.net_F(fake_y)    # G(X) -> F(G(X))
        
        
        fake_x = self.net_F(real_y)     # Y -> F(Y)
        recon_y = self.net_G(fake_x)    # F(Y) -> G(F(Y))
        
        return 
