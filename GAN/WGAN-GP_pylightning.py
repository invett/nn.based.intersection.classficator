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
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import kornia
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from dataloaders.sequencedataloader import txt_dataloader
from miscellaneous.utils import send_telegram_picture, send_telegram_message

import wandb
from PIL import Image


def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1.
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    min, max = feature_range
    x = x * (max - min) + min
    return x

class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a U-Net - 
    maps each pixel to a pixel with the correct number of output dimensions
    using a 1x1 convolution.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock: 
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x

    
class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels * 2)
        self.use_bn = use_bn
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock: 
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x

class Generator(nn.Module):
    def __init__(self, input_dim=100, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim

        self.gen = nn.Sequential(self.make_gen_block(input_dim, hidden_dim * 8, kernel_size=4, stride=1),
                                 self.make_gen_block(hidden_dim * 8, hidden_dim * 8, stride=1, padding=1),
                                 self.make_gen_block(hidden_dim * 8, hidden_dim * 8, padding=1),
                                 self.make_gen_block(hidden_dim * 8, hidden_dim * 4, padding=1),
                                 self.make_gen_block(hidden_dim * 4, hidden_dim * 4),
                                 self.make_gen_block(hidden_dim * 4, hidden_dim * 2),
                                 self.make_gen_block(hidden_dim * 2, hidden_dim),
                                 self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True))

        

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, padding=0, final_layer=False):
        """
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
            padding: padding...
        """
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding=padding, bias=False),
                nn.BatchNorm2d(output_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding=padding), nn.Tanh())

    def forward(self, noise):
        """
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, input_dim)
        """
        # GENERATOR
        x = noise.view(len(noise), self.input_dim, 1, 1)  # reshape vector in BxCxWxH
        imgs = self.gen(x)
        return imgs
    
    

class Patch_Discriminator(nn.Module):
    '''
    PatchGAN Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels=3, hidden_channels=8):
        super(Patch_Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)

    def forward(self, image):
        x0 = self.upfeature(image)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn
    
    
class Discriminator(nn.Module):
    """
    Discriminator Class
    """

    def __init__(self, im_chan=1, hidden_dim=64, apply_mask=False, image_type=''):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(self.make_disc_block(im_chan, hidden_dim, kernel_size=4),
                                  self.make_disc_block(hidden_dim, hidden_dim * 2),
                                  self.make_disc_block(hidden_dim * 2, hidden_dim * 4),
                                  self.make_disc_block(hidden_dim * 4, hidden_dim * 4),
                                  self.make_disc_block(hidden_dim * 4, hidden_dim * 8),
                                  self.make_disc_block(hidden_dim * 8, hidden_dim, stride=1, kernel_size=4),
                                  self.make_disc_block(hidden_dim, 1, final_layer=True))
        
        # load the mask
        if image_type == 'warping':
            mask = Image.open('GAN/MASK/alcala26_mask.png').convert('RGB')
        elif image_type == 'rgb':
            mask = Image.open('GAN/MASK/alcala26_mask_rgb.png').convert('RGB')
        elif apply_mask:
            exit(-1)

        mask = np.asarray(mask) / 255.0  # .transpose((2, 0, 1))
        self.mask = kornia.image_to_tensor(mask).cuda().half()
        self.apply_mask = apply_mask
        self.image_type = image_type

    def make_disc_block(self, input_channels, output_channels, kernel_size=3, stride=2, padding=0, final_layer=False):
        """
        Function to return a sequence of operations corresponding to a discriminator block of the DCGAN; 
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        """
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding, bias=False),
                nn.BatchNorm2d(output_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            return nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding))

    def forward(self, image):
        """
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        """
        # DISCRIMINATOR
        if self.apply_mask:
            image = image * self.mask
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)


class WGANGP(LightningModule):

    def __init__(self, latent_dim: int = 100, lr: float = 0.0002, b1: float = 0.5, b2: float = 0.999,
                 batch_size: int = 64, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        self.dataloader_choice = kwargs['dataloader']

        self.opt_g_frequency = kwargs['opt_g_frequency']
        self.opt_d_frequency = kwargs['opt_d_frequency']
        self.precision = kwargs['precision']  ## actually, it's not used, but with this we can debug in telegram routine
        self.decimate = kwargs['decimate']
        self.nowandb = kwargs['nowandb']
        self.loss = kwargs['loss']
        self.hidden_dim = kwargs['hidden_dim']
        self.image_type = kwargs['image_type']
        self.apply_mask = kwargs['apply_mask']
        self.label_smoothing = kwargs['label_smoothing']
        self.patch_disc = kwargs['patch_disc']

        # networks
        image_shape = (3, 224, 224)
        im_chan = 3
        self.generator = Generator(input_dim=latent_dim, im_chan=3, hidden_dim=self.hidden_dim)
        if self.patch_disc:
            self.discriminator = Patch_Discriminator(im_chan)
        else:
            self.discriminator = Discriminator(im_chan, hidden_dim=self.hidden_dim,
                                   apply_mask=self.apply_mask, image_type=self.image_type)
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)
        self.validation_z = torch.randn(9, self.latent_dim)

        self.example_input_array = torch.zeros(2, self.latent_dim)

    def weights_init(self, m):
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
        gradients = \
            torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake, create_graph=True,
                                retain_graph=True, only_inputs=True, )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.dataloader_choice == 'MNIST':
            imgs, _ = batch
        elif self.dataloader_choice == 'txt_dataloader':
            imgs = batch['data']
            imgs = scale(imgs)  # range [-1,1] to keep consistent with generated images (tanh activation function)
        else:
            return -1

        # sample noise (batch-size * size_of_latent_vector)
        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)

        lambda_gp = 10
        # For loss=BCELoss option
        criterion = torch.nn.BCEWithLogitsLoss()

        # print('optimizer_idx: ' + str(optimizer_idx))

        # train generator
        if optimizer_idx == 0:

            # generate images
            # self.generated_imgs = self(z)  # TODO: check: esto no entiendo para que sirve... en self.discriminator se llama de nuevo self(z)

            # log sampled images
            # sample_imgs = self.generated_imgs[:6]
            # grid = torchvision.utils.make_grid(sample_imgs)
            # self.logger.experiment.add_image('generated_images', grid, 0)

            fake_validity = self.discriminator(self(z))
            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones_like(fake_validity)
            if self.loss == 'wloss':
                # adversarial loss is binary cross-entropy
                g_loss = -torch.mean(fake_validity)
            else:
                # BCELoss (sigmoid activation function included)
                g_loss = criterion(fake_validity, valid)
            # tqdm_dict = {'g_loss': g_loss}
            # output = OrderedDict({'loss': g_loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict})
            # return output
            self.log('g_loss', g_loss, on_step=False, on_epoch=True)
            return g_loss

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        elif optimizer_idx == 1:
            fake_imgs = self(z)

            # Real images
            real_validity = self.discriminator(imgs)
            # Fake images
            fake_validity = self.discriminator(fake_imgs)
            if self.loss == 'wloss':
                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(imgs.data, fake_imgs.data)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            else:
                # put on GPU because we created this tensor inside training_loop
                real_valid = torch.ones_like(fake_validity) * np.random.uniform(low=0.7, high=1.2) if self.label_smoothing else torch.ones_like(fake_validity)
                fake_valid = torch.zeros_like(fake_validity) * np.random.uniform(low=0.0, high=0.3) if self.label_smoothing else torch.zeros_like(fake_validity)
                d_loss_fake = criterion(fake_validity, fake_valid)
                d_loss_real = criterion(real_validity, real_valid)  # torch.ones_like(real_validity)
                d_loss = (d_loss_fake + d_loss_real) / 2

            self.log('d_loss', d_loss, on_step=False, on_epoch=True)
            return d_loss

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return ({'optimizer': opt_g, 'frequency': self.opt_g_frequency},
                {'optimizer': opt_d, 'frequency': self.opt_d_frequency})

    def train_dataloader(self):
        if self.dataloader_choice == 'MNIST':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), ])
            dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
            return DataLoader(dataset, batch_size=self.batch_size)

        if self.dataloader_choice == 'txt_dataloader':

            if self.image_type == 'warping':
                train_path = '/home/ballardini/DualBiSeNet/alcala-26.01.2021_selected_warped/prefix_all.txt'
            elif self.image_type == 'rgb':
                train_path = '/home/ballardini/DualBiSeNet/alcala-26.01.2021_selected/prefix_all.txt'
            else:
                print('dataloader error')
                exit(-1)

            #rgb_image_test_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
            #                                                transforms.Normalize((0.485, 0.456, 0.406),
            #                                                                     (0.229, 0.224, 0.225))])

            # for GANS, normalize with these values https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
            rgb_image_test_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            dataset_ = txt_dataloader(train_path, transform=rgb_image_test_transforms, decimateStep=self.decimate)
            dataloader_ = DataLoader(dataset_, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True)
            return dataloader_

    def on_epoch_end(self):
        z = self.validation_z.to(self.device)

        # log sampled images
        sample_imgs = self(z)

        # image_tensor = (image_tensor + 1) / 2
        # image_unflat = image_tensor.detach().cpu()
        # image_grid = make_grid(image_unflat[:num_images], nrow=nrow)

        # send single image .. ensure 32bit images for telegram, otherwise BUMMM
        data = kornia.tensor_to_image(sample_imgs[0]).astype(np.float32)  # will be between -1 and 1

        from_max = 1.0
        from_min = -1.0
        to_max = 1.
        to_min = 0.
        a = (to_max - to_min) / (from_max - from_min)
        b = to_max - a * from_max
        data_ = np.array([(a * x + b) for x in data])
        label = 'GAN - SINGLE IMAGE\ncurrent epoch: ' + str(
            self.current_epoch)  # TODO wandb --> self.logger.name , see https://docs.wandb.ai/integrations/lightning
        a = plt.figure()
        plt.imshow(data_)
        send_telegram_picture(a, label)
        plt.close('all')

        # send grid
        grid = torchvision.utils.make_grid(sample_imgs, nrow=3)
        data = kornia.tensor_to_image(grid).astype(np.float32)  # will be between -1 and 1
        from_max = 1.0
        from_min = -1.0
        to_max = 1.
        to_min = 0.
        a = (to_max - to_min) / (from_max - from_min)
        b = to_max - a * from_max
        data_ = np.array([(a * x + b) for x in data])
        label = 'GAN - GRID\ncurrent epoch: ' + str(self.current_epoch)
        a = plt.figure()
        plt.imshow(data_)
        send_telegram_picture(a, label)
        plt.close('all')

        # send grid to wandb
        if not self.nowandb:
            data = kornia.tensor_to_image(grid).astype(np.float32)  # will be between -1 and 1
            from_max = 1.0
            from_min = -1.0
            to_max = 1.
            to_min = 0.
            a = (to_max - to_min) / (from_max - from_min)
            b = to_max - a * from_max
            data_ = np.array([(a * x + b) for x in data])
            label = 'GAN - GRID\ncurrent epoch: ' + str(self.current_epoch)
            a = plt.figure()
            plt.imshow(data_)
            self.trainer.logger.experiment.log(
                {"current grid": wandb.Image(plt, caption=f"Epoch:{self.current_epoch}")})
            plt.close('all')

        # self.logger.experiment.add_image('generated_images', grid, self.current_epoch)


def main(args: Namespace) -> None:
    # keep track of parameters in logs
    print(args)

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = WGANGP(**vars(args))

    if args.wandb_group_id:
        group_id = args.wandb_group_id
    else:
        group_id = 'GENERIC-GAN'

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # If use distubuted training  PyTorch recommends to use DistributedDataParallel.
    # See: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel

    if not args.nowandb:
        run = wandb.init(project='GAN')
        run.save()
        wandb_logger = WandbLogger(project='GAN', entity='chiringuito', group=group_id, job_type="training")
        wandb_logger.watch(model)
        # saves a file like: ./trainedmodels/GAN/wandb_run_id-epoch=100.ckpt
        checkpoint_callback = ModelCheckpoint(dirpath='./trainedmodels/GAN/',
                                              filename=os.path.join(run.id, '-{epoch:02d}.ckpt'), monitor='g_loss',
                                              mode='min')
        if args.resume_from_checkpoint == 'no':
            trainer = Trainer(gpus=args.gpus, logger=wandb_logger, weights_summary='full', precision=args.precision,
                              profiler=True, callbacks=[checkpoint_callback], max_epochs=args.max_epochs)
        else:
            trainer = Trainer(gpus=args.gpus, logger=wandb_logger, weights_summary='full', precision=args.precision,
                              profiler=True, callbacks=[checkpoint_callback], max_epochs=args.max_epochs,
                              resume_from_checkpoint=args.resume_from_checkpoint)

    else:
        checkpoint_callback = ModelCheckpoint(dirpath='./trainedmodels/GAN/',
                                              filename=os.path.join('nowandb-{epoch:02d}.ckpt'), monitor='g_loss',
                                              mode='min')
        trainer = Trainer(gpus=args.gpus, weights_summary='full', precision=args.precision, profiler=True,
                          callbacks=[checkpoint_callback])

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="number of GPUs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--hidden_dim", type=int, default=64, help="channels width multiplier")
    parser.add_argument("--opt_g_frequency", type=int, default=1, help="generator frequency")
    parser.add_argument("--opt_d_frequency", type=int, default=1, help="discriminator frequency")
    parser.add_argument('--dataloader', type=str, default='txt_dataloader',
                        choices=['fromAANETandDualBisenet', 'generatedDataset', 'Kitti2011_RGB', 'triplet_OBB',
                                 'triplet_BOO', 'triplet_ROO', 'triplet_ROO_360', 'triplet_3DOO_360', 'Kitti360',
                                 'Kitti360_3D', 'txt_dataloader', 'lstm_txt_dataloader', 'MNIST'],
                        help='One of the supported datasets')

    parser.add_argument('--wandb_group_id', type=str, help='Set group id for the wandb experiment')
    parser.add_argument('--nowandb', action='store_true', help='use this flag to DISABLE wandb logging')
    parser.add_argument("--precision", type=int, default=32, help="32 or 16 bit precision", choices=[32, 16])
    parser.add_argument("--loss", type=str, default='bce', help="Choose loss between Wasserstein or BCE",
                        choices=['wloss', 'bce'])
    parser.add_argument('--decimate', type=int, default=1, help='How much of the points will remain after '
                                                                'decimation')

    parser.add_argument("--max_epochs", type=int, default=10000, help="max number of epochs")
    parser.add_argument("--image_type", type=str, default='warping', help="Choose between warping or rgb",
                        choices=['rgb', 'warping'])
    parser.add_argument("--resume_from_checkpoint", type=str, default='no', help="absolute path for checkpoint resume")
    parser.add_argument('--apply_mask', action='store_true', help='apply mask to the generated imgs')
    parser.add_argument('--label_smoothing', action='store_true', help='apply label smoothing')    
    parser.add_argument('--patch_disc', action='store_true', help='use PatchGAN Discriminator')

    hparams = parser.parse_args()

    main(hparams)
