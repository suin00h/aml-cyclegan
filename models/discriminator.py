import torch
from torch import nn

class Discriminator(nn.Module):
    """
    'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70x70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.
    convolution based discriminator: 3x128x128 -> 1x1x1
    """
    def __init__(
        self,
        lch=64
    ):
        super().__init__()
        
        self.conv1 = self.get_conv_block(in_ch=3, out_ch=lch, is_initial=True)
        self.conv2 = self.get_conv_block(in_ch=lch, out_ch=lch*2, is_initial=True)
        self.conv3 = self.get_conv_block(in_ch=lch*2, out_ch=lch*4, is_initial=True)
        self.conv4 = self.get_conv_block(in_ch=lch*4, out_ch=lch*4, is_initial=True)
        self.conv5 = nn.Conv2d(lch*4, 1, 4, 1, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        return x
    
    def get_conv_block(self, in_ch, out_ch, stride=2, is_initial=False):
        conv = nn.Conv2d(in_ch, out_ch, 4, stride, 1)
        norm = nn.InstanceNorm2d(out_ch)
        relu = nn.LeakyReLU(0.2, True)
        
        if is_initial:
            block = [conv, relu]
        else:
            block = [conv, norm, relu]
        
        return nn.Sequential(*block)

if __name__ == "__main__":
    net = Discriminator().to("cuda")
    
    test = torch.randn((10, 3, 128, 128), device="cuda")
    
    out = net(test)
    
    print(out.shape)
