import itertools

import torch

from models.cyclegan import CycleGAN
from models.generator import Generator
from models.discriminator import Discriminator

import argparse, os
from dataset import TrainDataset
from torch.utils.data import DataLoader
from configs.config import TrainConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, default='maps')
    return parser.parse_args()

if __name__ == "__main__":
    lr = 2e-5
    lambda_x = 2e-4
    lambda_y = 2e-4
    lambda_idt = 2e-4
    lambda_clip = 0.2
    
    args = parse_args()
    yaml_path = os.path.join("configs", f"{args.name}.yaml")
    params = TrainConfig(yaml_path)
    dataset = TrainDataset(params)
    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)
    
    net_G = Generator().to("cuda")
    net_F = Generator().to("cuda")
    net_Dx = Discriminator().to("cuda")
    net_Dy = Discriminator().to("cuda")
    
    net = CycleGAN(
        net_G, net_F, net_Dx, net_Dy,
        lambda_x, lambda_y, lambda_idt, lambda_clip,
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
    # real_x = torch.randn((10, 3, 128, 128), device="cuda")
    # real_y = torch.randn((10, 3, 128, 128), device="cuda")
    data = next(iter(dataloader))
    real_x, real_y = data
    
    # forward
    fake_x, fake_y, recon_x, recon_y, idt_x, idt_y = net(real_x, real_y)
    
    # update Generator
    opt_Gen.zero_grad()
    loss_Gen= net.get_generator_loss(
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
    
    from torchvision.utils import save_image
    import torchvision.transforms.functional as TF
    
    real_A = real_x[0].detach().cpu()
    real_B = real_y[0].detach().cpu()
    fake_B = fake_y[0].detach().cpu()
    fake_A = fake_x[0].detach().cpu()
    recon_A = recon_x[0].detach().cpu()
    recon_B = recon_y[0].detach().cpu()

    # [real_A | fake_B | recon_A], [real_B | fake_A | recon_B]
    image_grid_A = torch.cat([real_A, fake_B, recon_A], dim=-1)
    image_grid_B = torch.cat([real_B, fake_A, recon_B], dim=-1)
    image_grid = torch.cat([image_grid_A, image_grid_B], dim=1)
    
    print(fake_A.shape)
    print(fake_A.max())
    print(fake_A.min())
    print(real_A.max())
    print(real_A.min())
    
    save_image(image_grid, 'out.png', normalize=True)