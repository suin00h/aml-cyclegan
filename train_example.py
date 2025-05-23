import itertools

import torch

from models.cyclegan import CycleGAN
from models.generator import Generator
from models.discriminator import Discriminator

if __name__ == "__main__":
    lr = 2e-5
    lambda_x = 2e-4
    lambda_y = 2e-4
    lambda_idt = 2e-4
    
    net_G = Generator().to("cuda")
    net_F = Generator().to("cuda")
    net_Dx = Discriminator().to("cuda")
    net_Dy = Discriminator().to("cuda")
    
    net = CycleGAN(
        net_G, net_F, net_Dx, net_Dy,
        lambda_x, lambda_y, lambda_idt,
        "cuda"
    )
    
    opt_Gen = torch.optim.Adam(
        itertools.chain(net_G.parameters(), net_F.parameters()),
        lr=lr
    )
    opt_Disc = torch.optim.Adam(
        itertools.chain(net_Dx.parameters(), net_Dy.parameters()),
        lr=lr
    )
    
    # preparing input real images
    # images have to be normalized in advance: (-1, 1)
    real_x = torch.randn((10, 3, 128, 128), device="cuda")
    real_y = torch.randn((10, 3, 128, 128), device="cuda")
    
    # forward
    fake_x, fake_y, recon_x, recon_y, idt_x, idt_y = net(real_x, real_y)
    
    # update Generator
    opt_Gen.zero_grad()
    loss_Gen = net.get_generator_loss(
        real_x, real_y,
        fake_x, fake_y,
        recon_x, recon_y,
        idt_x, idt_y
    )
    loss_Gen.backward()
    opt_Gen.step()
    
    # update Discriminator
    opt_Disc.zero_grad()
    loss_Disc = net.get_discriminator_loss(
        real_x, real_y,
        fake_x, fake_y
    )
    loss_Disc.backward()
    opt_Disc.step()