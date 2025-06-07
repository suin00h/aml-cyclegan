import os
import torch
import wandb
import torch.nn.init as init
from torch.optim import lr_scheduler

def get_scheduler(params, optimizer):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + params.epoch_cnt - params.n_epochs) / float(params.n_epochs_decay + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler

def init_wandb(params):
    wandb.init(
        project="CycleGAN",
        name=params.dataname,
        config={
            "epochs": params.n_epochs + params.n_epochs_decay,
            "lr": params.lr,
            "lambda_x": params.lambda_x,
            "lambda_y": params.lambda_y,
            "lambda_idt": params.lambda_idt
        }
    )

def init_weights(net, init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

def load_model_weights(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found at {path}")
    model.load_state_dict(torch.load(path, map_location="cuda"))
    print(f"Loaded weights from {path}")