import os
import itertools
import torch
import argparse
import wandb

from torch.utils.data import DataLoader

from models.cyclegan import CycleGAN
from models.generator import Generator
from models.discriminator import Discriminator

from configs.config import TrainConfig
from dataset import TrainDataset
from utils.buffer import ImageBuffer
from utils.setup import init_wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, default='map')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    yaml_path = os.path.join("configs", f"{args.name}.yaml")

    params = TrainConfig(yaml_path)
    dataset = TrainDataset(params)
    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)

    if params.max_buffer > 0:
        fake_x_pool = ImageBuffer(params.max_buffer)
        fake_y_pool = ImageBuffer(params.max_buffer)

    init_wandb(params)

    net_G = Generator().to("cuda")
    net_F = Generator().to("cuda")
    net_Dx = Discriminator().to("cuda")
    net_Dy = Discriminator().to("cuda")
    
    net = CycleGAN(
        net_G, net_F, net_Dx, net_Dy,
        params.lambda_x, params.lambda_y, params.lambda_idt, params.lambda_clip,
        "cuda"
    )

    net.init_all_weights()
    
    opt_Gen = torch.optim.Adam(
        itertools.chain(net_G.parameters(), net_F.parameters()),
        lr=params.lr
    )
    opt_Disc = torch.optim.Adam(
        itertools.chain(net_Dx.parameters(), net_Dy.parameters()),
        lr=params.lr
    )

    # setup learning rate
    net.set_scheduler(params, [opt_Gen, opt_Disc])
    
    for epoch in range(params.epoch_cnt, params.n_epochs + params.n_epochs_decay + 1):
        net.update_lr()
        for i, data in enumerate(dataloader):
            real_x, real_y = data

            # forward
            fake_x, fake_y, recon_x, recon_y, idt_x, idt_y = net(real_x, real_y)
            
            # update Generator
            opt_Gen.zero_grad()
            loss_dict = net.get_generator_loss(
                real_x, real_y,
                fake_x, fake_y,
                recon_x, recon_y,
                idt_x, idt_y
            )
            loss_dict['total'].backward()
            opt_Gen.step()
            
            # update Discriminator
            opt_Disc.zero_grad()

            if params.max_buffer > 0:
                # TODO: improve buffer algorithm
                fake_x_buffered = fake_x_pool.sample(fake_x.detach())
                fake_y_buffered = fake_y_pool.sample(fake_y.detach())

                loss_Disc = net.get_discriminator_loss(
                    real_x, real_y,
                    fake_x_buffered, fake_y_buffered
                )
            else:
                loss_Disc = net.get_discriminator_loss(
                    real_x, real_y,
                    fake_x, fake_y
                )

            loss_Disc.backward()
            opt_Disc.step()

            if i % 10 == 0:  # every 10 steps
                current_lr = opt_Gen.param_groups[0]['lr']
                wandb.log({
                    "Generator Loss (Total)": loss_dict['total'].item(),
                    "Generator Loss / GAN": loss_dict['gan'].item(),
                    "Generator Loss / Cycle": loss_dict['cycle'].item(),
                    "Generator Loss / Identity": loss_dict['identity'].item(),
                    "Discriminator Loss": loss_Disc.item(),
                    "Learning Rate": current_lr,
                    "Epoch": epoch,
                    "Step": i,
                })

            if i == 0:
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
                
                image_grid = (((image_grid + 1) / 2.0) * 255).clamp(0, 255).byte()

                wandb.log({
                    f"Epoch {epoch} | Generated Samples": wandb.Image(image_grid, caption=f"Epoch {epoch}")
                })

        if epoch % params.save_epochs == 0:
            net.save_model(params.save_dir, epoch)
