import os
import torch
import torch.nn.functional as F

from torch import nn
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

from models.generator import Generator
from models.discriminator import Discriminator

from utils.setup import get_scheduler, init_weights

class CycleGAN(nn.Module):
    def __init__(
        self,
        net_G: "Generator",
        net_F: "Generator",
        net_Dx: "Discriminator",
        net_Dy: "Discriminator",
        lambda_x,
        lambda_y,
        lambda_idt,
        lambda_clip,
        device: "str"
    ):
        super().__init__()

        self.net_G = net_G          # maps X to Y
        self.net_F = net_F          # maps Y to X
        self.net_Dx = net_Dx        # distinguish between images x and translated images F(y)
        self.net_Dy = net_Dy        # distinguish between images y and translated images G(x)
        self.clip = CLIPFeatureExtractor()

        self.lambda_x = lambda_x             # regularizing constant
        self.lambda_y = lambda_y
        self.lambda_idt = lambda_idt
        self.lambda_clip = lambda_clip

        self.crit_idt = nn.L1Loss()
        self.crit_gan = GANLoss("vanilla").to(device)
        self.crit_cycle = nn.L1Loss()

    def forward(self, real_x, real_y):
        """
        Input:      Two real images of size (B, C, H, W) from distribution X and Y respectively.
        """
        fake_y = self.net_G(real_x)     # X -> G(X)
        recon_x = self.net_F(fake_y)    # G(X) -> F(G(X))

        fake_x = self.net_F(real_y)     # Y -> F(Y)
        recon_y = self.net_G(fake_x)    # F(Y) -> G(F(Y))

        idt_y = self.net_G(real_y)      # G(y) \sim y
        idt_x = self.net_F(real_x)      # F(x) \sim x

        return fake_x, fake_y, recon_x, recon_y, idt_x, idt_y

    def get_generator_loss(
        self,
        real_x, real_y,
        fake_x, fake_y,
        recon_x, recon_y,
        idt_x, idt_y,
    ):
        self.set_requires_grad([self.net_Dx, self.net_Dy], False)  # Ds require no gradients when optimizing Gs

        # compute identity loss
        loss_idt_x = self.crit_idt(idt_x, real_x) * self.lambda_x   # ||F(x) - x|| * lambda_x * lambda_idt
        loss_idt_y = self.crit_idt(idt_y, real_y) * self.lambda_y   # ||G(y) - y|| * lambda_y * lambda_idt
        loss_idt = (loss_idt_x + loss_idt_y) * self.lambda_idt

        # compute GAN loss
        loss_gan_x = self.crit_gan(self.net_Dx(fake_x), target_is_real=True)    # Criterion(Dx(F(y), 1))
        loss_gan_y = self.crit_gan(self.net_Dy(fake_y), target_is_real=True)    # Criterion(Dy(G(x), 1))
        loss_gan = loss_gan_x + loss_gan_y

        # compute cycle loss
        loss_cycle_x = self.crit_cycle(recon_x, real_x) * self.lambda_x # ||F(G(x)) - x|| * lambda_x
        loss_cycle_y = self.crit_cycle(recon_y, real_y) * self.lambda_y # ||G(F(y)) - y|| * lambda_y
        loss_cycle = loss_cycle_x + loss_cycle_y

        # compute CLIP loss
        loss_clip_x = self.crit_clip(self.clip(real_x), self.clip(fake_x))
        loss_clip_y = self.crit_clip(self.clip(real_y), self.clip(fake_y))
        loss_clip = (loss_clip_x + loss_clip_y) * self.lambda_clip

        # total loss: GAN loss + Cycle loss + Identity loss
        return loss_idt + loss_gan + loss_cycle + loss_clip

    def get_discriminator_loss(
        self,
        real_x, real_y,
        fake_x, fake_y
    ):
        self.set_requires_grad([self.net_Dx, self.net_Dy], True)

        # compute discriminator loss on real data
        loss_real_x = self.crit_gan(self.net_Dx(real_x), target_is_real=True)
        loss_real_y = self.crit_gan(self.net_Dy(real_y), target_is_real=True)
        loss_real = loss_real_x + loss_real_y

        # compute discriminator loss on fake data
        loss_fake_x = self.crit_gan(self.net_Dx(fake_x.detach()), target_is_real=False)
        loss_fake_y = self.crit_gan(self.net_Dy(fake_y.detach()), target_is_real=False)
        loss_fake = loss_fake_x + loss_fake_y

        return (loss_real + loss_fake) * 0.5

    def crit_clip(self, emb1, emb2):
        sim = F.cosine_similarity(emb1, emb2, dim=1)
        return 1 - sim

    # from original code repo
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_scheduler(self, params, optimizers):
        self.schedulers = [get_scheduler(params, optimizer) for optimizer in optimizers]

    def update_lr(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def init_all_weights(self, init_gain=0.02):
        init_weights(self.net_G, init_gain)
        init_weights(self.net_F, init_gain)
        init_weights(self.net_Dx, init_gain)
        init_weights(self.net_Dy, init_gain)

    def save_model(self, save_dir, epoch):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        torch.save(self.net_G.state_dict(), os.path.join(save_dir, f"net_G_epoch_{epoch}.pth"))
        torch.save(self.net_F.state_dict(), os.path.join(save_dir, f"net_F_epoch_{epoch}.pth"))
        torch.save(self.net_Dx.state_dict(), os.path.join(save_dir, f"net_Dx_epoch_{epoch}.pth"))
        torch.save(self.net_Dy.state_dict(), os.path.join(save_dir, f"net_Dy_epoch_{epoch}.pth"))

# from original code repo
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - typically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class CLIPFeatureExtractor(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()
    
    def forward(self, img):
        img = ((img + 1) / 2).clamp(0, 1) # NCHW; [-1, 1] to [0, 1]
        img = (img * 255).to(torch.uint8)
        
        imgs = [transforms.ToPILImage()(im.cpu()) for im in img]
        inputs = self.processor(images=imgs, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            features = self.model.get_image_features(**inputs) # (N, 512)
            
        return features
