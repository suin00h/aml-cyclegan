import os
import itertools
import torch
import argparse

from models.cyclegan import CycleGAN
from models.generator import Generator
from models.discriminator import Discriminator

from configs.config import TrainConfig
from dataset import TrainDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, default='map')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    yaml_path = os.path.join("configs", f"{args.name}.yaml")

    params = TrainConfig(yaml_path)
    dataset = TrainDataset(params)

    net_G = Generator().to("cuda")
    net_F = Generator().to("cuda")
    net_Dx = Discriminator().to("cuda")
    net_Dy = Discriminator().to("cuda")
    
    net = CycleGAN(
        net_G, net_F, net_Dx, net_Dy,
        params.lambda_x, params.lambda_y, params.lambda_idt,
        "cuda"
    )
    
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
        for i, data in enumerate(dataset):
            real_x, real_y = data
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