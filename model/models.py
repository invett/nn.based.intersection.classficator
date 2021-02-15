import os

import torch
from torchvision import models


class Resnet18(torch.nn.Module):

    def __init__(self, pretrained=True, embeddings=False, num_classes=None, freeze=False):
        super().__init__()
        self.embeddings = embeddings

        model = models.resnet18(pretrained=pretrained)
        if freeze:
            for param in model.parameters():
                param.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

        if not embeddings:
            self.fc = torch.nn.Linear(512, num_classes)

    def forward(self, data):
        x = self.conv1(data)
        x = self.relu(self.bn1(x))
        x = self.maxpool(x)
        feature1 = self.layer1(x)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)
        embedding = self.avgpool(feature4)
        if self.embeddings:
            return embedding
        else:
            prediction = self.fc(embedding)
            return prediction


class Vgg11(torch.nn.Module):

    def __init__(self, pretrained=True, embeddings=False, num_classes=None, freeze=False):
        super().__init__()
        model = models.vgg11(pretrained=pretrained)
        self.embeddings = embeddings

        if freeze:
            for param in model.parameters():
                param.requires_grad = False

        self.features = model.features
        self.avgpool = model.avgpool

        if embeddings:
            self.classifier = model.classifier
            self.classifier[6] = torch.nn.Linear(4096, 512)
        else:
            self.classifier = model.classifier
            self.classifier[6] = torch.nn.Linear(4096, num_classes)

    def forward(self, data):
        features = self.features(data)
        avg = self.avgpool(features)
        prediction = self.classifier(avg)

        return prediction


class LSTM(torch.nn.Module):

    def __init__(self, num_classes, lstm_dropout, fc_dropout, input_size=512):
        super().__init__()

        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=256, num_layers=2, batch_first=True,
                                  dropout=lstm_dropout)
        self.fc = torch.nn.Linear(256, num_classes)
        self.drop = torch.nn.Dropout(p=fc_dropout)

    def forward(self, data):
        output, (hn, _) = self.lstm(data)  # --> hn shape (layers x batch x 256)
        last_hidden = hn[-1]  # -->(batch, 256)
        prediction = self.fc(self.drop(last_hidden))

        return prediction


class Resnet50_Coco(torch.nn.Module):  # Resnet50 trained in coco segmentation dataset

    def __init__(self, embeddings_size=512):
        super().__init__()
        self.embeddings_size = embeddings_size  # If embeddings size is 512 the fc should be trained

        model = models.segmentation.fcn_resnet50(pretrained=True, progress=True, num_classes=21, aux_loss=None)

        for param in model.parameters():  # Freeze encoder parameters
            param.requires_grad = False

        self.encoder = model.backbone
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if embeddings_size == 512:
            self.fc = torch.nn.Linear(2048, 512)

    def forward(self, data):
        x = self.encoder(data)
        x = self.avgpool(x['out'])
        if self.embeddings_size == 512:
            embedding = self.fc(x)
        else:
            embedding = x

        return embedding
