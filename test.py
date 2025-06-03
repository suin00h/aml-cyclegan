import os
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.generator import Generator
from configs.config import TestConfig
from dataset import TrainDataset
from utils.setup import load_model_weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, default='map')
    parser.add_argument('--epoch', type=int, required=True, help='epoch number to load')
    return parser.parse_args()


@torch.no_grad()
def run_inference(net_G, net_F, dataloader, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    net_G.eval()
    net_F.eval()

    for idx, (real_x, real_y) in enumerate(dataloader):
        real_x = real_x.to("cuda")
        real_y = real_y.to("cuda")

        fake_y = net_G(real_x)
        fake_x = net_F(real_y)

        # Save original and translated images
        for b in range(real_x.size(0)):
            x = real_x[b].cpu()
            y = real_y[b].cpu()
            fx = fake_x[b].cpu()
            fy = fake_y[b].cpu()

            # save [real_x | fake_y]
            save_image(torch.cat([x, fy], dim=-1), os.path.join(save_dir, f"AtoB_{idx}_{b}.png"), normalize=True)
            # save [real_y | fake_x]
            save_image(torch.cat([y, fx], dim=-1), os.path.join(save_dir, f"BtoA_{idx}_{b}.png"), normalize=True)


if __name__ == "__main__":
    args = parse_args()
    yaml_path = os.path.join("configs", f"{args.name}.yaml")
    params = TestConfig(yaml_path)

    test_dataset = TrainDataset(params)  # assuming same dataset class is used
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Define Generators
    net_G = Generator().to("cuda")
    net_F = Generator().to("cuda")

    # Load weights
    G_path = os.path.join(params.save_dir, f"net_G_epoch_{args.epoch}.pth")
    F_path = os.path.join(params.save_dir, f"net_F_epoch_{args.epoch}.pth")

    load_model_weights(net_G, G_path)
    load_model_weights(net_F, F_path)

    # Run inference
    save_result_dir = os.path.join("results", args.name, f"epoch_{args.epoch}")
    run_inference(net_G, net_F, test_dataloader, save_result_dir)