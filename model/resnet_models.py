import torch
from torchvision import models

from torch import nn

"""
2048 > 1024
2048 > 1024
1024 > 512
512 > 256
256 > num_classes

"""


def first_convBlock(in_channels, out_channels, kernel_size, stride, padding):
    convblock = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

    return convblock


def convBlock(in_channels, out_channels, kernel_size, stride=1, padding=1):
    convblock = nn.Sequential(
        nn.Dropout2d(p=0.15),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.GroupNorm(4, out_channels),
        nn.ReLU()
    )

    return convblock


def set_classifier(num_class):
    classifier = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(256, num_class)
    )

    return classifier


def set_feature_vector(features, avgpool):
    classifier = nn.Sequential(
        features,
        avgpool,
        nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False),  # --> 512 x 4 x 4
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(4, stride=1),  # --> 512 x 1 x 1
    )

    return classifier


def get_model_vgg(model_version, num_classes=7, embedding=False, pretrained=True):
    if model_version == 'vgg11':
        model = models.vgg11_bn(pretrained=pretrained)
        if embedding:
            model = set_feature_vector(model.features, model.avgpool)
        else:
            model.classifier[6] = torch.nn.Linear(4096, num_classes)

    if model_version == 'vgg13':
        model = models.vgg13_bn(pretrained=pretrained)
        if embedding:
            model = set_feature_vector(model.features, model.avgpool)
        else:
            model.classifier[6] = torch.nn.Linear(4096, num_classes)

    if model_version == 'vgg16':
        model = models.vgg16_bn(pretrained=pretrained)
        if embedding:
            model = set_feature_vector(model.features, model.avgpool)
        else:
            model.classifier[6] = torch.nn.Linear(4096, num_classes)

    if model_version == 'vgg19':
        model = models.vgg19_bn(pretrained=pretrained)
        if embedding:
            model = set_feature_vector(model.features, model.avgpool)
        else:
            model.classifier[6] = torch.nn.Linear(4096, num_classes)

    return model


def get_model_resnet(model_version, num_classes, transfer=False, pretrained=True, greyscale=False, embedding=False):
    if model_version == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        if transfer:
            for param in model.parameters():
                param.requires_grad = False
        if greyscale:
            model.conv1 = torch.nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)  # Change first layer to 1 channel
        if embedding:
            model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Delete fc layer
        else:
            model.fc = torch.nn.Linear(512, num_classes)

    elif model_version == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        if transfer:
            for param in model.parameters():
                param.requires_grad = False
        if greyscale:
            model.conv1 = torch.nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)  # Change first layer to 1 channel
        if embedding:
            model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Delete fc layer
        else:
            model.fc = torch.nn.Linear(512, num_classes)

    elif model_version == 'resnet50':
        model = models.resnet34(pretrained=pretrained)
        if transfer:
            for param in model.parameters():
                param.requires_grad = False
        fc = set_classifier(num_classes)
        model.fc = fc

    elif model_version == 'resnet101':
        model = models.resnet34(pretrained=pretrained)
        if transfer:
            for param in model.parameters():
                param.requires_grad = False
        fc = set_classifier(num_classes)
        model.fc = fc

    elif model_version == 'resnet152':
        model = models.resnet34(pretrained=pretrained)
        if transfer:
            for param in model.parameters():
                param.requires_grad = False
        fc = set_classifier(num_classes)
        model.fc = fc

    return model


def get_model_resnext(model_version, num_classes, transfer, pretrained):
    if model_version == 'resnext50':
        model = models.resnext50_32x4d(pretrained=pretrained)
        if transfer:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = torch.nn.Linear(2048, num_classes)
    elif model_version == 'resnext101':
        model = models.resnext101_32x8d(pretrained=pretrained)
        if transfer:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = torch.nn.Linear(2048, num_classes)

    return model


class Personalized(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.convBlock_1 = first_convBlock(3, 64, 7, 2, 3)
        self.convBlock_2 = convBlock(64, 128, 3)
        self.convBlock_3 = convBlock(128, 256, 3)
        self.convBlock_4 = convBlock(256, 256, 7, 7)

        self.fc = nn.Linear(256, 128)
        self.bn = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)

        # functionals
        self.ReLu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2)

    def forward(self, data):
        x = self.convBlock_1(data)
        x = self.pool(x)
        x = self.convBlock_2(x)
        x = self.pool(x)
        x = self.convBlock_3(x)
        x = self.pool(x)
        x = self.convBlock_4(x)
        x = self.avgpool(x)
        x = x.view(-1, 256)
        x = self.ReLu(self.bn(self.fc(x)))
        x = self.dropout(x)
        out = self.classifier(x)

        return out


class Personalized_small(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.convBlock_1 = first_convBlock(3, 64, 7, 2, 3)
        self.convBlock_2 = convBlock(64, 128, 3)

        self.fc1 = nn.Linear(25088, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.classifier = nn.Linear(1024, num_classes)

        # functionals
        self.ReLu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)

    def forward(self, data):
        x = self.convBlock_1(data)  # B x 64 x 112 x 112
        x = self.pool(x)  # B x 64 x 56 x 56
        x = self.convBlock_2(x)  # B x 128 x 56 x 56
        x = self.pool(x)  # B x 128 x 28 x 28
        x = self.avgpool(x)  # B x 128 x 14 x 14
        x = x.view(-1, 25088)
        x = self.ReLu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.ReLu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        out = self.classifier(x)

        return out
