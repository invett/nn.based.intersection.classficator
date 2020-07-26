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
        nn.Linear(256, num_classes)
    )

    return classifier


def get_model_resnet(model_version, num_classes, transfer):
    if model_version == 'resnet18':
        model = models.resnet18(pretrained=True)
        if transfer:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = torch.nn.Linear(512, num_classes)
    elif model_version == 'resnet34':
        model = models.resnet34(pretrained=True)
        if transfer:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = torch.nn.Linear(512, num_classes)
    elif model_version == 'resnet50':
        model = models.resnet34(pretrained=True)
        if transfer:
            for param in model.parameters():
                param.requires_grad = False
        fc = set_classifier(num_classes)
        model.fc = fc
    elif model_version == 'resnet101':
        model = models.resnet34(pretrained=True)
        if transfer:
            for param in model.parameters():
                param.requires_grad = False
        fc = set_classifier(num_classes)
        model.fc = fc
    elif model_version == 'resnet152':
        model = models.resnet34(pretrained=True)
        if transfer:
            for param in model.parameters():
                param.requires_grad = False
        fc = set_classifier(num_classes)
        model.fc = fc

    return model


def get_model_resnext(model_version, num_classes, transfer):
    if model_version == 'resnext50':
        model = models.resnext50_32x4d(pretrained=True)
        if transfer:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = torch.nn.Linear(2048, num_classes)
    elif model_version == 'resnext101':
        model = models.resnext101_32x8d(pretrained=True)
        if transfer:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = torch.nn.Linear(2048, num_classes)

    return model
