import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

import numpy as np

import warnings

warnings.filterwarnings("ignore")

from dataloaders.sequencedataloader import txt_dataloader
from miscellaneous.utils import send_telegram_picture, send_telegram_message

torch.manual_seed(0)


def show_tensor_images(image_tensor, num_images=25, nrow=5, show=False, type='Fake'):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    send_telegram_picture(plt, "Type: " + str(type))
    if show:
        plt.show()
    plt.close()


def make_grad_hook():
    '''
    Function to keep track of gradients for visualization purposes, 
    which fills the grads list when using model.apply(grad_hook).
    '''
    grads = []

    def grad_hook(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            grads.append(m.weight.grad)

    return grads, grad_hook


class Generator(nn.Module):
    '''
    Generator Class
    '''

    def __init__(self, input_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        # Build the neural network
        self.gen = nn.Sequential(self.make_gen_block(input_dim, hidden_dim * 2),
                                 self.make_gen_block(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=1),
                                 self.make_gen_block(hidden_dim * 4, hidden_dim * 8),
                                 self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4),
                                 self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, padding=1),
                                 self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=4, padding=1),
                                 self.make_gen_block(hidden_dim, im_chan, kernel_size=4, padding=1, final_layer=True))

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

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, input_dim)
        '''
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)


def get_noise(n_samples, input_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, input_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        input_dim: the dimension of the input vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, input_dim, device=device)


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


def get_input_dimensions(z_dim, shape, n_classes):
    '''
    Function for getting the size of the conditional input dimensions 
    from z_dim, the image shape, and number of classes.
    Parameters:
        z_dim: the dimension of the noise vector, a scalar
        mnist_shape: the shape of each MNIST image as (C, W, H), which is (1, 28, 28)
        n_classes: the total number of classes in the dataset
    Returns: 
        generator_input_dim: the input dimensionality of the conditional generator, 
                          which takes the noise and class vectors
        discriminator_im_chan: the number of input channels to the discriminator
    '''
    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = shape[0] + n_classes
    return generator_input_dim, discriminator_im_chan


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


#### GRADIENT PENALTY ######


def get_gradient(crit, real, fake, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)

    # Take the gradient of the scores with respect to the images
    gradient = \
        torch.autograd.grad(inputs=mixed_images, outputs=mixed_scores, grad_outputs=torch.ones_like(mixed_scores),
                            create_graph=True, retain_graph=True, )[0]
    return gradient


def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)

    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean((gradient_norm - torch.ones_like(gradient_norm)) ** 2)
    return penalty


#############  WGAN LOSSES ###############

def get_gen_loss(crit_fake_pred):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    gen_loss = -torch.mean(crit_fake_pred)
    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty 
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + gp * c_lambda
    return crit_loss


#######################################################################################################################
##### Main ####


n_epochs = 1000
z_dim = 128
display_step = 150
batch_size = 64
lr = 0.0005
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10
crit_repeats = 5
device = 'cuda'

n_classes = 7
dataset_shape = (3, 224, 224)

generator_input_dim, discriminator_im_chan = get_input_dimensions(z_dim, dataset_shape, n_classes)

gen = Generator(input_dim=generator_input_dim, im_chan=3).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator(im_chan=discriminator_im_chan).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

#######################################################################################################################
# generative chiringuito

decimate = 1

train_path = '/home/ballardini/DualBiSeNet/alcala-26.01.2021_selected_warped/prefix_all.txt'

rgb_image_train_transforms = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.RandomAffine(15, translate=(0.0, 0.1), shear=(-5, 5)),
     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

rgb_image_test_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_dataset = txt_dataloader(train_path, transform=rgb_image_test_transforms, decimateStep=decimate)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

cur_step = 0
generator_losses = []
discriminator_losses = []
noise_and_labels = False
fake = False
fake_image_and_labels = False
real_image_and_labels = False
disc_fake_pred = False
disc_real_pred = False

send_telegram_message("Starting GAN training")

for epoch in range(n_epochs):

    tq = tqdm.tqdm(total=len(dataloader) * batch_size)
    tq.set_description('epoch %d, lr %.e' % (epoch, lr))

    # for real, labels in dataloader:
    for sample in dataloader:
        real = sample['data']
        labels = sample['label']

        cur_batch_size = len(real)
        real = real.to(device)

        one_hot_labels = get_one_hot_labels(labels.to(device), n_classes)
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, dataset_shape[1], dataset_shape[2])

        mean_iteration_disc_loss = 0
        for _ in range(crit_repeats):
            ### Update critic ###
            disc_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
            fake = gen(noise_and_labels)
            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
            real_image_and_labels = combine_vectors(real, image_one_hot_labels)
            disc_fake_pred = disc(fake_image_and_labels)
            disc_real_pred = disc(real_image_and_labels)

            epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
            gradient = get_gradient(disc, real_image_and_labels, fake_image_and_labels.detach(), epsilon)
            gp = gradient_penalty(gradient)
            disc_loss = get_crit_loss(disc_fake_pred, disc_real_pred, gp, c_lambda)

            # Keep track of the average critic loss in this batch
            mean_iteration_disc_loss += disc_loss.item() / crit_repeats
            # Update gradients
            disc_loss.backward(retain_graph=True)
            # Update optimizer
            disc_opt.step()
        discriminator_losses += [mean_iteration_disc_loss]

        ### Update generator ###
        gen_opt.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
        noise_and_labels_2 = combine_vectors(fake_noise_2, one_hot_labels)
        fake_2 = gen(noise_and_labels_2)
        fake_image_and_labels_2 = combine_vectors(fake_2, image_one_hot_labels)
        disc_fake_pred = disc(fake_image_and_labels_2)
        gen_loss = get_gen_loss(disc_fake_pred)
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the generator losses
        generator_losses += [gen_loss.item()]

        ### Visualization code ###
        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            disc_mean = sum(discriminator_losses[-display_step:]) / display_step
            show_tensor_images(fake, type='Fake')
            show_tensor_images(real, type='True')
            step_bins = 20
            x_axis = sorted([i * step_bins for i in range(len(generator_losses) // step_bins)] * step_bins)
            num_examples = (len(generator_losses) // step_bins) * step_bins
            plt.plot(range(num_examples // step_bins),
                     torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1), label="Generator Loss")
            plt.plot(range(num_examples // step_bins),
                     torch.Tensor(discriminator_losses[:num_examples]).view(-1, step_bins).mean(1),
                     label="Discriminator Loss")
            plt.legend()
            send_telegram_picture(plt,
                                  "Step: " + str(cur_step) +
                                  "\nGenerator loss: " + str(gen_mean) +
                                  "\nDiscriminator loss: " + str(disc_mean))
            plt.close()

        cur_step += 1
        tq.update(cur_batch_size)

send_telegram_message("GAN training finished")
