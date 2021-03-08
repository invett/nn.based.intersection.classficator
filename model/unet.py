import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision
from torch.utils.data import DataLoader

from dataloader import VGGDataLoader
import os

from skimage import io

import datetime
import atexit

# U-Net: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/pdf/1505.04597.pdf

class uNet(nn.Module):

    def __init__(self):
        super(uNet, self).__init__()

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d( 3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            )   #POOL_1 : 64

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            )   #POOL_2 : 128

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            )   #POOL_3:  256

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            )   #POOL_4: 512


    def forward(self, x):
        conv1_noMaxPool = self.conv_block1(x)
        conv1 = F.max_pool2d(conv1_noMaxPool, kernel_size=2, stride=2, ceil_mode=True)

        conv2_noMaxPool = self.conv_block2(conv1)
        conv2 = F.max_pool2d(conv2_noMaxPool, kernel_size=2, stride=2, ceil_mode=True)

        conv3_noMaxPool = self.conv_block3(conv2)
        conv3 = F.max_pool2d(conv3_noMaxPool, kernel_size=2, stride=2, ceil_mode=True)

        conv4_noMaxPool = self.conv_block4(conv3)
        conv4 = F.max_pool2d(conv4_noMaxPool, kernel_size=2, stride=2, ceil_mode=True)

        return conv1,conv2,conv3,conv4,conv1_noMaxPool,conv2_noMaxPool,conv3_noMaxPool,conv4_noMaxPool

########################################################################################################################

class SegmentationNetwork_Unet(nn.Module):
    def __init__(self,numeroClassi):
        super(SegmentationNetwork_Unet, self).__init__()

        self.feature_extractor = uNet()

        #nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.convLayer_1 = nn.Conv2d( 512,        1024,kernel_size=3,padding=1)
        self.convLayer_2 = nn.Conv2d(1024,         512,kernel_size=3,padding=1)

        #in_channels, out_channels, kernel_size, stride = 1,padding = 0, output_padding = 0, groups = 1, bias = True, dilation = 1
        self.deConv_1    = nn.ConvTranspose2d(512, 512,kernel_size=2,padding=0,stride=2)
        self.deConv_2    = nn.ConvTranspose2d(256, 256,kernel_size=2,padding=0,stride=2)
        self.deConv_3    = nn.ConvTranspose2d(128, 128,kernel_size=2,padding=0,stride=2)
        self.deConv_4    = nn.ConvTranspose2d( 64,  64,kernel_size=2,padding=0,stride=2)

        self.conv_dx_layer_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv_dx_layer_2 = nn.Conv2d( 512, 256, kernel_size=3, padding=1)
        self.conv_dx_layer_3 = nn.Conv2d( 512, 256, kernel_size=3, padding=1)
        self.conv_dx_layer_4 = nn.Conv2d( 256, 128, kernel_size=3, padding=1)
        self.conv_dx_layer_5 = nn.Conv2d( 256, 128, kernel_size=3, padding=1)
        self.conv_dx_layer_6 = nn.Conv2d( 128,  64, kernel_size=3, padding=1)

        self.conv_dx_layer_7 = nn.Conv2d( 128, 64, kernel_size=3, padding=1)
        self.conv_dx_layer_8 = nn.Conv2d(  64, 64, kernel_size=3, padding=1)

        self.classifier      = nn.Conv2d(  64, numeroClassi, kernel_size=1, padding=0)


    def forward(self, x):

        conv1,conv2,conv3,conv4,conv1_noMaxPool,conv2_noMaxPool,conv3_noMaxPool,conv4_noMaxPool = self.feature_extractor(x)  #qui chiama il forward (di uNet)

        bottom_layer_1 = self.convLayer_1(conv4)
        bottom_layer_1 = F.relu(bottom_layer_1)

        bottom_layer_2 = self.convLayer_2(bottom_layer_1)
        bottom_layer_2: object = F.relu(bottom_layer_2)

        up_layer_1              = self.deConv_1(bottom_layer_2)
        up_layer_1_concatenated = torch.cat((conv4_noMaxPool,up_layer_1),dim=1)
        up_layer_1_concatenated = F.relu(self.conv_dx_layer_1(up_layer_1_concatenated))
        up_layer_1_concatenated = F.relu(self.conv_dx_layer_2(up_layer_1_concatenated))

        up_layer_2              = self.deConv_2(up_layer_1_concatenated)
        up_layer_2_concatenated = torch.cat((conv3_noMaxPool,up_layer_2),dim=1)
        up_layer_2_concatenated = F.relu(self.conv_dx_layer_3(up_layer_2_concatenated))
        up_layer_2_concatenated = F.relu(self.conv_dx_layer_4(up_layer_2_concatenated))

        up_layer_3              = self.deConv_3(up_layer_2_concatenated)
        up_layer_3_concatenated = torch.cat((conv2_noMaxPool,up_layer_3),dim=1)
        up_layer_3_concatenated = F.relu(self.conv_dx_layer_5(up_layer_3_concatenated))
        up_layer_3_concatenated = F.relu(self.conv_dx_layer_6(up_layer_3_concatenated))

        up_layer_4              = self.deConv_4(up_layer_3_concatenated)
        up_layer_4_concatenated = torch.cat((conv1_noMaxPool,up_layer_4),dim=1)
        up_layer_4_concatenated = F.relu(self.conv_dx_layer_7(up_layer_4_concatenated))
        up_layer_4_concatenated = F.relu(self.conv_dx_layer_8(up_layer_4_concatenated))

        final_classification      = self.classifier(up_layer_4_concatenated)

        return final_classification

########################################################################################################################

def goodbye(filename_rete):
    global net, optimizer, epoch, save
    if save:
        print('Final save, using filename: ' , filename_rete)
        torch.save({'state_dict': net.state_dict(), 'state_optimizer': optimizer.state_dict(), 'epoch': epoch},
                   filename_rete)
        print('Save Ok!')

def main():
    global net, optimizer, epoch, save

    filename_rete = './checkpoint__unet_1.pth'
    atexit.register(goodbye, filename_rete)

    load = 1
    save = 1
    visualize = 0

    scale_factor = 32   #FCN32
    scale_factor = 16   #FNC16
    scale_factor = 1    #FCN8

    dataset_training   = VGGDataLoader('/media/RAIDONE/DATASETS/KITTI_SEMANTIC_LOPEZ','Training', downsample_size=scale_factor)
    dataset_validation = VGGDataLoader('/media/RAIDONE/DATASETS/KITTI_SEMANTIC_LOPEZ','Validation', downsample_size=scale_factor)

    pytorchDataLoader_training   = DataLoader(dataset_training,   batch_size = 3, shuffle = True, num_workers = 2, pin_memory = True,
                                   drop_last = True, timeout = 0, worker_init_fn = None)
    pytorchDataLoader_validation = DataLoader(dataset_validation, batch_size = 1, shuffle = False, num_workers = 2, pin_memory = True,
                                   drop_last = True, timeout = 0, worker_init_fn = None)

    loss_function = nn.CrossEntropyLoss(ignore_index=255) #la classe di default per IGNORARE. La tolgo anche più avanti in visualizzazione! #REF:01


    numeroclassi = 11
    #net = SegmentationNetwork_FCN32s(numeroclassi).cuda() #downsample_size=32
    #net = SegmentationNetwork_FCN16s(numeroclassi).cuda() #downsample_size=16
    #net = SegmentationNetwork_FCN8s(numeroclassi).cuda()   #downsample_size=8
    net = SegmentationNetwork_Unet(numeroclassi).cuda()

    if load:
        if os.path.exists(filename_rete):
            loaded_net = torch.load(filename_rete, map_location='cpu')
            net.load_state_dict(loaded_net['state_dict'])

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[100,200,500],gamma=0.5)

    epochs = 2000
    for epoch in range(0,epochs):

        scheduler.step(epoch)
        training_loss = 0.0
        equals = 0.0

        before = datetime.datetime.now()
        net.train()
        for batch_index, sample in enumerate(pytorchDataLoader_training):

            img , label = sample
            img = img.cuda()
            label = label.cuda()

            optimizer.zero_grad()
            #output_unet, output_upsampled = net(img)
            output_unet = net(img)

            #output = output.permute(0,2,3,1).contiguous()
            #output = output.view(-1,11)
            #label = label.view(-1)

            loss = loss_function(output_unet, label)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            if visualize:
                label[label==255]=12 # 255 è la label UNKNOWN, togliamola --- #REF:01
                io.imshow(F.upsample_bilinear(label.unsqueeze(1).float(), scale_factor=scale_factor)[0, 0].cpu().numpy())
                io.show()
                #io.imshow(output_upsampled[0].argmax(0).cpu().numpy())
                #io.show()

        torch.cuda.synchronize()

        now = datetime.datetime.now()
        later = datetime.datetime.now()
        elapsed = later - before
        print(f'{now.strftime("%Y-%m-%d %H:%M")} --- Epoch {epoch} / {epochs}    Training Loss: {training_loss/batch_index} \tElapsed: {round(elapsed.microseconds / 1000)} ms')

        validation_loss = 0.0

        before = datetime.datetime.now()
        net.eval()
        for batch_index, sample in enumerate(pytorchDataLoader_validation):
            img , label = sample
            img = img.cuda()
            label = label.cuda()

            with torch.no_grad():
                #output_unet, output_upsampled  = net(img)
                output_unet = net(img)

            loss = loss_function(output_unet, label)
            validation_loss += loss.item()

            #if visualize:
                #io.imshow(output_upsampled[0].argmax(0).cpu().numpy())
                #io.show()


        torch.cuda.synchronize()

        later = datetime.datetime.now()
        now = datetime.datetime.now()
        elapsed = later - before
        print(f'{now.strftime("%Y-%m-%d %H:%M")} --- Epoch {epoch} / {epochs}    Validation Loss: {validation_loss/batch_index} \tElapsed: {round(elapsed.microseconds / 1000)} ms')

        if save:
            before  = datetime.datetime.now()
            torch.save({'state_dict':net.state_dict(),'state_optimizer':optimizer.state_dict(),'epoch':epoch}, filename_rete)
            later   = datetime.datetime.now()
            elapsed = later - before
            print(f'Elapsed time (saving model only): {round(elapsed.microseconds / 1000)} ms')

    return

if __name__ == '__main__':
    main()