import torch
from torchvision import models


def get_model_resnet(model_version):
    if model_version == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, 7)
    elif model_version == 'resnet34':
        model = models.resnet34(pretrained=True)
        model.fc = torch.nn.Linear(512, 7)
    elif model_version == 'resnet50':
        model = models.resnet34(pretrained=True)
        model.fc = torch.nn.Linear(2048, 7)
    elif model_version == 'resnet101':
        model = models.resnet34(pretrained=True)
        model.fc = torch.nn.Linear(2048, 7)
    elif model_version == 'resnet152':
        model = models.resnet34(pretrained=True)
        model.fc = torch.nn.Linear(2048, 7)

    return model


def get_model_resnext(model_version):
    if model_version == 'resnet18':
        model = models.resnext50(pretrained=True)
        model.fc = torch.nn.Linear(2048, 7)
    elif model_version == 'resnet34':
        model = models.resnext101(pretrained=True)
        model.fc = torch.nn.Linear(2048, 7)

    return model
