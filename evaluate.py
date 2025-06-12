import os
import argparse
import json
import torch
import lpips

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_fid import fid_score
from torchvision.models.inception import inception_v3

import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# ---------------- Argument ----------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True, default=200)
    parser.add_argument('--size', nargs=2, type=int, default=[256, 256],
                        help="Image size as two integers, e.g. --size 256 256")
    return parser.parse_args()

# ---------------- Dataset ----------------

class ResultDataset(Dataset):
    def __init__(self, folder_path, size):
        self.paths = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.endswith(('.jpg', '.png'))]
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

# IS

def calculate_inception_score(imgs, model, splits=10):
    imgs = imgs.cuda()
    with torch.no_grad():
        preds = model(imgs)
        preds = F.softmax(preds, dim=1).cpu().numpy()

    scores = []
    N = preds.shape[0]
    for i in range(splits):
        part = preds[i * N // splits: (i + 1) * N // splits]
        py = np.mean(part, axis=0)
        scores.append(np.exp(np.mean([entropy(p, py) for p in part])))

    return np.mean(scores), np.std(scores)

def calculate_is(fake_dir, size, model):
    dataset = ResultDataset(fake_dir, size)
    all_images = [dataset[i] for i in range(len(dataset))]
    all_images = torch.stack(all_images, dim=0).cuda()

    return calculate_inception_score(all_images, model)

# FID

def calculate_fid(real_dir, fake_dir):
    return fid_score.calculate_fid_given_paths([real_dir, fake_dir],
                                                batch_size=1,
                                                device='cuda',
                                                dims=2048)

# SSIM / PSNR / LPIPS

def evaluate_pairwise_metrics(fake_dir, real_dir, size, loss_fn):
    ssim_list, psnr_list, lpips_list = [], [], []

    # convert into [-1, 1]
    transform_lpips = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    # convert into [0, 1]
    transform_std = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    files = sorted(os.listdir(fake_dir))
    for fname in files:
        if not fname.endswith(('.jpg', '.png')):
            continue

        fake_img = Image.open(os.path.join(fake_dir, fname)).convert('RGB')
        real_img = Image.open(os.path.join(real_dir, fname)).convert('RGB')

        fake_lpips = transform_lpips(fake_img).unsqueeze(0).cuda()
        real_lpips = transform_lpips(real_img).unsqueeze(0).cuda()

        fake_std = transform_std(fake_img).detach().cpu().numpy()
        real_std = transform_std(real_img).detach().cpu().numpy()

        lpips_score = loss_fn(fake_lpips, real_lpips).item()
        ssim = compare_ssim(fake_std.transpose(1,2,0),
                            real_std.transpose(1,2,0),
                            channel_axis=-1,
                            data_range=1.0,
                            win_size=min(7, min(fake_std.shape[1], fake_std.shape[2])))
        psnr = compare_psnr(real_std, fake_std, data_range=1.0)
        
        lpips_list.append(lpips_score)
        ssim_list.append(ssim)
        psnr_list.append(psnr)

    return {
        "SSIM": float(np.mean(ssim_list)),
        "PSNR": float(np.mean(psnr_list)),
        "LPIPS": float(np.mean(lpips_list))
    }


if __name__ == "__main__":
    args = parse_args()
    size = tuple(args.size)

    fake_x_dir = f"results/{args.name}/epoch_{args.epoch}/fake_x"
    fake_y_dir = f"results/{args.name}/epoch_{args.epoch}/fake_y"
    real_x_dir = f"results/{args.name}/epoch_{args.epoch}/real_x"
    real_y_dir = f"results/{args.name}/epoch_{args.epoch}/real_y"

    # Load models once
    inception_model = inception_v3(pretrained=True, transform_input=False).eval().cuda()
    lpips_loss_fn = lpips.LPIPS(net='alex').cuda()

    results = {}

    # FID
    results["FID_x"] = float(calculate_fid(real_x_dir, fake_x_dir))
    results["FID_y"] = float(calculate_fid(real_y_dir, fake_y_dir))

    # IS
    is_x_mean, is_x_std = calculate_is(fake_x_dir, size, inception_model)
    is_y_mean, is_y_std = calculate_is(fake_y_dir, size, inception_model)
    results["IS_x"] = {"mean": float(is_x_mean), "std": float(is_x_std)}
    results["IS_y"] = {"mean": float(is_y_mean), "std": float(is_y_std)}

    # PSNR, SSIM, LPIPS
    results["Metrics_x"] = evaluate_pairwise_metrics(fake_x_dir, real_x_dir, size, lpips_loss_fn)
    results["Metrics_y"] = evaluate_pairwise_metrics(fake_y_dir, real_y_dir, size, lpips_loss_fn)

    # Save to JSON
    out_file = f"results/{args.name}/epoch_{args.epoch}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"✅ Metrics saved to {out_file}")