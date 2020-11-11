import os

import torch
from torchvision import models


class resnet18(torch.nn.Module):

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


class vgg11(torch.nn.Module):  # A lo mejor no funciona con las redes que ya estan pre entrenadas...

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


class LSTM(resnet18, vgg11, torch.nn.Module):

    def __init__(self, features_model, num_classes, pretrained_path=None):
        super().__init__()
        if features_model == 'resnet':
            resnet18.__init__(pretrained=True, embeddings=True, freeze=True)
        if features_model == 'vgg':
            vgg11.__init__(pretrained=True, embeddings=True, freeze=True)

        if os.path.isfile(pretrained_path):
            print("=> loading checkpoint '{}'".format(pretrained_path))
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if features_model == 'resnet':
                resnet18.load_state_dict(checkpoint['model_state_dict'])
            if features_model == 'vgg':
                vgg11.load_state_dict(checkpoint['model_state_dict'])
            print("=> loaded checkpoint '{}'".format(pretrained_path))

        if features_model == 'resnet':
            resnet18.eval()
        if features_model == 'vgg':
            vgg11.eval()

        self.lstm = torch.nn.LSTM(input_size=512, hidden_size=256, batch_first=True)
        self.fc = torch.nn.Linear(256, num_classes)

    def forward(self, data):
        if self.features_model == 'resnet':
            features = resnet18.forward(data)

        if self.features_model == 'vgg':
            features = vgg11.forward(data)
        output, (hn, cn) = self.lstm(features)
        prediction = self.fc(hn)

        return prediction
