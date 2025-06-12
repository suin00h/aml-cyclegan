import os
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.generator import Generator
from configs.config import TestConfig
from dataset import TestDataset
from utils.setup import load_model_weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, default='map')
    parser.add_argument('--epoch', type=int, required=True, help='epoch number to load')
    return parser.parse_args()


@torch.no_grad()
def run_inference(net_G, net_F, dataloader, save_dir, params):
    # Separate folders for each image type
    os.makedirs(save_dir, exist_ok=True)

    subdirs = ["real_x", "fake_y", "real_y", "fake_x"]
    for sub in subdirs:
        os.makedirs(os.path.join(save_dir, sub), exist_ok=True)

    net_G.eval()
    net_F.eval()

    for idx, batch in enumerate(dataloader):
        if params.dataname == "cityscapes":
            real_x, real_y, filename = batch
        else:
            real_x, real_y = batch
            filename = f"{idx}"

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

            name_tag = filename[b] if isinstance(filename, (list, tuple)) else filename
            
            # Save each image separately
            if params.dataname == "cityscapes":
                save_image(x, os.path.join(save_dir, "real_x", f"{name_tag}_gtFine.png"), normalize=True)
                save_image(fy, os.path.join(save_dir, "fake_y", f"{name_tag}_leftImg8bit.png"), normalize=True)
                save_image(y, os.path.join(save_dir, "real_y", f"{name_tag}_leftImg8bit.png"), normalize=True)
                save_image(fx, os.path.join(save_dir, "fake_x", f"{name_tag}_gtFine.png"), normalize=True)
            else:
                save_image(x, os.path.join(save_dir, "real_x", f"{name_tag}_{b}.png"), normalize=True)
                save_image(fy, os.path.join(save_dir, "fake_y", f"{name_tag}_{b}.png"), normalize=True)
                save_image(y, os.path.join(save_dir, "real_y", f"{name_tag}_{b}.png"), normalize=True)
                save_image(fx, os.path.join(save_dir, "fake_x", f"{name_tag}_{b}.png"), normalize=True)

if __name__ == "__main__":
    args = parse_args()
    yaml_path = os.path.join("configs", f"{args.name}.yaml")
    params = TestConfig(yaml_path)

    test_dataset = TestDataset(params)  # assuming same dataset class is used
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
    if params.dataname == "cityscapes":
        save_result_dir = os.path.join("results", args.name, f"fcn/epoch_{args.epoch}")
    else:
        save_result_dir = os.path.join("results", args.name, f"epoch_{args.epoch}")
    run_inference(net_G, net_F, test_dataloader, save_result_dir, params)