import argparse
import os
import numpy as np
import math
import sys
from tqdm import tqdm

import torchvision.transforms as transforms
import torchvision.utils as vutils

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from network import *
from dataloader import MyDataset


if __name__ == '__main__':
    nc = 3
    ndf = 64
    ngf = 64
    ngpu = 1
    nz = 100
    batch_size = 128
    num_workers = 4
    lr = 2e-4
    max_epochs = 200
    clip_value = 0.01
    n_critic = 5
    seed = 999
    sample_interval = 200

    save_dir = 'log/images'

    try:
        os.makedirs(save_dir)
    except:
        pass

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = MyDataset("training_data_cartoonset10k_96_96.npy", batch_size)    
    dataloader = dataset.get_loader()

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')  

    # Initialize generator and discriminator
    netG = Generator(nc, ngf, nz, ngpu).to(device)
    netD = Discriminator(nc, ndf, ngpu).to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Optimizers
    optimizer_G = torch.optim.RMSprop(netG.parameters(), lr=lr)
    optimizer_D = torch.optim.RMSprop(netD.parameters(), lr=lr)


    # ----------
    #  Training
    # ----------
    batches_done = 0
    for epoch in tqdm(range(max_epochs)):
        for i, data in enumerate(dataloader):

            # Configure input
            real_imgs = data.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = torch.randn(batch_size, nz, 1, 1, device=device)

            # Generate a batch of images
            fake_imgs = netG(z).detach()

            # Adversarial loss
            loss_D = -torch.mean(netD(real_imgs)) + torch.mean(netD(fake_imgs))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in netD.parameters():
                p.data.clamp_(-clip_value, clip_value)

            # Train the generator every n_critic iterations
            if i % n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = netG(z)
                # Adversarial loss
                loss_G = -torch.mean(netD(gen_imgs))

                loss_G.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, max_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
                )

            if batches_done % sample_interval == 0:
                vutils.save_image(gen_imgs.data[:25], f"{save_dir}/{batches_done}.png", nrow=5, normalize=True)
            batches_done += 1
