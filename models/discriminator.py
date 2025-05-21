from torch import nn


class Discriminator(nn.Module):
    """
    'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70x70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.
    """
    pass
