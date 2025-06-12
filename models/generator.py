import torch
from torch import nn

class Generator(nn.Module):
    """
    U-Net: [unet_128] (for 128x128 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597
    unet-based i2i generator: 3x128x128 -> ... -> 512x1x1 -> ... -> 3x128x128
    """
    def __init__(
        self,
        lch=64
    ):
        super().__init__()
        
        unet_submodule = UnetBlock(out_ch=lch*8, in_ch=lch*8, innermost=True)
        
        unet_submodule = UnetBlock(out_ch=lch*8, in_ch=lch*8, submodule=unet_submodule)
        unet_submodule = UnetBlock(out_ch=lch*8, in_ch=lch*8, submodule=unet_submodule)
        
        unet_submodule = UnetBlock(out_ch=lch*4, in_ch=lch*8, submodule=unet_submodule)
        unet_submodule = UnetBlock(out_ch=lch*2, in_ch=lch*4, submodule=unet_submodule)
        unet_submodule = UnetBlock(out_ch=lch, in_ch=lch*2, submodule=unet_submodule)
        
        self.unet = UnetBlock(out_ch=3, in_ch=lch, submodule=unet_submodule, outermost=True)
    
    def forward(self, x):
        x = self.unet(x)
        
        return x

class UnetBlock(nn.Module):
    """
    Covering given submodule with down/upsample blocks
    """
    def __init__(
        self,
        out_ch: int,
        in_ch: int,
        submodule=None,
        outermost=False,
        innermost=False
    ):
        super().__init__()
        self.out_ch = out_ch
        self.in_ch = in_ch
        
        self.outermost = outermost
        self.innermost = innermost
        
        self.down_block = self.get_down_block()
        self.submodule = submodule
        self.up_block = self.get_up_block()
    
    def forward(self, x):
        x_skip = x
        x = self.down_block(x)
        if self.submodule is not None:
            x = self.submodule(x)
        x = self.up_block(x)
        
        if not self.outermost:
            x = torch.cat([x, x_skip], dim=1)
        return x
    
    def get_down_block(self):
        relu = nn.LeakyReLU(0.2, True)
        conv = nn.Conv2d(self.out_ch, self.in_ch, kernel_size=4,
                         stride=2, padding=1)
        norm = nn.InstanceNorm2d(self.in_ch)
        
        if self.outermost:
            block = [conv]
        elif self.innermost:
            block = [relu, conv]
        else:
            block = [relu, conv, norm]
        return nn.Sequential(*block)
    
    def get_up_block(self):
        relu = nn.LeakyReLU(0.2, True)
        norm = nn.InstanceNorm2d(self.out_ch)
        if self.outermost:
            conv = nn.ConvTranspose2d(self.in_ch * 2, self.out_ch, kernel_size=4,
                                      stride=2, padding=1)
            block = [relu, conv, nn.Tanh()]
        elif self.innermost:
            conv = nn.ConvTranspose2d(self.in_ch, self.out_ch, kernel_size=4,
                                      stride=2, padding=1)
            block = [relu, conv, norm]
        else:
            conv = nn.ConvTranspose2d(self.in_ch * 2, self.out_ch, kernel_size=4,
                                      stride=2, padding=1)
            block = [relu, conv, norm]
        return nn.Sequential(*block)

if __name__ == "__main__":
    net = Generator().to("cuda")
    
    test = torch.randn((10, 3, 128, 128), device="cuda")
    
    out = net(test)
    
    print(out.shape)