"""
python wgan_gp.py
After a few epochs, launch TensorBoard to see the images being generated at every batch:
tensorboard --logdir default
"""
import os

from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer


class Generator(nn.Module):
    def __init__(self, input_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim

        def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, padding=0, final_layer=False):
            '''
            Function to return a sequence of operations corresponding to a generator block of DCGAN;
            a transposed convolution, a batchnorm (except in the final layer), and an activation.
            Parameters:
                input_channels: how many channels the input feature representation has
                output_channels: how many channels the output feature representation should have
                kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
                stride: the stride of the convolution
                final_layer: a boolean, true if it is the final layer and false otherwise 
                          (affects activation and batchnorm)
            '''
            if not final_layer:
                return nn.Sequential(
                    nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding=padding),
                    nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True))
            else:
                return nn.Sequential(
                    nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding=padding), nn.Tanh())
        
        
        self.gen = nn.Sequential(
                      self.make_gen_block(input_dim, hidden_dim * 2),
                      self.make_gen_block(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=1),
                      self.make_gen_block(hidden_dim * 4, hidden_dim * 8),
                      self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4),
                      self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, padding=1),
                      self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=4, padding=1),
                      self.make_gen_block(hidden_dim, im_chan, kernel_size=4, padding=1, final_layer=True)
                   )


    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, input_dim)
        '''
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)



class Discriminator(nn.Module):
    '''
    Discriminator Class
    '''
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(self.make_disc_block(im_chan, hidden_dim, kernel_size=4),
                                  self.make_disc_block(hidden_dim, hidden_dim * 2),
                                  self.make_disc_block(hidden_dim * 2, hidden_dim * 2),
                                  self.make_disc_block(hidden_dim * 2, hidden_dim * 4),
                                  self.make_disc_block(hidden_dim * 4, hidden_dim * 4, padding=1),
                                  self.make_disc_block(hidden_dim * 4, hidden_dim * 2, padding=1),
                                  self.make_disc_block(hidden_dim * 2, hidden_dim, kernel_size=4, padding=1),
                                  self.make_disc_block(hidden_dim, 1, kernel_size=4, padding=1, final_layer=True))

    def make_disc_block(self, input_channels, output_channels, kernel_size=3, stride=2, padding=0, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a discriminator block of the DCGAN; 
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding),
                                 nn.BatchNorm2d(output_channels), nn.LeakyReLU(0.2, inplace=True))
        else:
            return nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding))

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)


class WGANGP(LightningModule):

    def __init__(self,
                 latent_dim: int = 100,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = 64, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size

        # networks
        self.image_shape = (3, 224, 224)
        im_chan = 3
        self.n_classes = 7
        generator_input_dim = latent_dim + n_classes
        discriminator_im_chan = im_chan + n_classes
        self.generator = Generator(input_dim=generator_input_dim, im_chan=3)
        self.discriminator = Discriminator(im_chan=discriminator_im_chan)
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.validation_z = torch.randn(8, self.latent_dim)

        self.example_input_array = torch.zeros(2, self.latent_dim)
    
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, z):
        return self.generator(z)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
  
    
    def get_one_hot_labels(labels, n_classes):
        '''
        Function for creating one-hot vectors for the labels.
        Parameters:
            labels: tensor of labels from the dataloader
            n_classes: the total number of classes in the dataset
        '''
        return torch.nn.functional.one_hot(labels, n_classes)


    def combine_vectors(x, y):
        combined = (torch.cat((x.float(), y.float()), 1))
        return combined
    

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, labels = batch

        # sample noise
        z = torch.randn(real.shape[0], self.latent_dim)
        z = z.type_as(real)
        
        one_hot_labels = self.get_one_hot_labels(labels, self.n_classes)
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, self.image_shape[1], self.image_shape[2])
        lambda_gp = 10

        # train generator
        if optimizer_idx == 0:
            # concatenate noise and labels
            noise_and_labels = self.combine_vectors(z, one_hot_labels)
            # generate images
            self.generated_imgs = self(noise_and_labels)
            fake_image_and_labels = self.combine_vectors(self.generated_imgs, image_one_hot_labels)
            
            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = -torch.mean(self.discriminator(fake_image_and_labels)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        elif optimizer_idx == 1:
            noise_and_labels = self.combine_vectors(z, one_hot_labels)
            #generate images
            fake_imgs = self(noise_and_labels)
            fake_image_and_labels = self.combine_vectors(fake_imgs, image_one_hot_labels)
            real_image_and_labels = combine_vectors(real, image_one_hot_labels)
            # Real images
            real_validity = self.discriminator(real_image_and_labels)
            # Fake images
            fake_validity = self.discriminator(fake_image_and_labels)
            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty(real.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        n_critic = 5

        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return (
            {'optimizer': opt_g, 'frequency': 1},
            {'optimizer': opt_d, 'frequency': n_critic}
        )

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size)

    def on_epoch_end(self):
        z = self.validation_z.to(self.device)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)


def main(args: Namespace) -> None:
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = WGANGP(**vars(args))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # If use distubuted training  PyTorch recommends to use DistributedDataParallel.
    # See: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
    trainer = Trainer(gpus=args.gpus)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0, help="number of GPUs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")

    hparams = parser.parse_args()

    main(hparams)
