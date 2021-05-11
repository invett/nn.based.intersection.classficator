import os

import torch
from torchvision import models


class Resnet(torch.nn.Module):

    def __init__(self, pretrained=True, embeddings=False, num_classes=None, version='resnet18'):
        super().__init__()
        self.embeddings = embeddings
        self.version = version

        if version == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        if version == 'resnet34':
            model = models.resnet18(pretrained=pretrained)
        if version == 'resnet50':
            model = models.resnet18(pretrained=pretrained)
        if version == 'resnet101':
            model = models.resnet18(pretrained=pretrained)
        if version == 'resnet152':
            model = models.resnet18(pretrained=pretrained)

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
            if version == 'resnet18' or version == 'resnet34':
                self.fc = torch.nn.Linear(512, num_classes)
            else:
                self.fc = torch.nn.Linear(2048, num_classes)
        else:
            if not (version == 'resnet18' or version == 'resnet34'):
                self.reducer = torch.nn.Linear(2048, 512)

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
            if self.version == 'resnet18' or self.version == 'resnet34':
                return embedding.squeeze()
            else:
                embedding = self.reducer(embedding)
                return embedding.squeeze()
        else:
            prediction = self.fc(embedding)
            return prediction


class VGG(torch.nn.Module):

    def __init__(self, pretrained=True, embeddings=False, num_classes=None, version='vgg11'):
        super().__init__()
        if version == 'vgg11':
            model = models.vgg11_bn(pretrained=pretrained)
        if version == 'vgg13':
            model = models.vgg13_bn(pretrained=pretrained)
        if version == 'vgg16':
            model = models.vgg16_bn(pretrained=pretrained)
        if version == 'vgg19':
            model = models.vgg19_bn(pretrained=pretrained)

        self.embeddings = embeddings

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
        avg = torch.flatten(avg, start_dim=1)
        prediction = self.classifier(avg)

        return prediction


class Mobilenet_v3(torch.nn.Module):

    def __init__(self, pretrained=True, embeddings=False, num_classes=None, version='mobilenet_v3_small'):
        super().__init__()

        self.embeddings = embeddings
        self.version = version

        if version == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(pretrained=pretrained)
        if version == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(pretrained=pretrained)

        self.features = model.features
        self.avgpool = model.avgpool

        if embeddings:
            self.classifier = model.classifier
            if version == 'small':
                self.classifier[3] = torch.nn.Linear(1024, 512)
            else:
                self.classifier[3] = torch.nn.Linear(1280, 512)
        else:
            self.classifier = model.classifier
            if version == 'small':
                self.classifier[3] = torch.nn.Linear(1024, num_classes)
            else:
                self.classifier[3] = torch.nn.Linear(1280, num_classes)

    def forward(self, data):
        features = self.features(data)
        avg = self.avgpool(features)
        prediction = self.classifier(avg)

        return prediction


class Inception_v3(torch.nn.Module):

    def __init__(self, pretrained=True, embeddings=False, num_classes=None):
        super().__init__()

        self.embeddings = embeddings

        self.model = models.inception_v3(pretrained=pretrained)

        if embeddings:
            self.model.fc = torch.nn.Linear(2048, 512)
            self.model.AuxLogits.fc = torch.nn.Linear(2048, 512)
        else:
            self.model.fc = torch.nn.Linear(2048, num_classes)
            self.model.AuxLogits.fc = torch.nn.Linear(2048, num_classes)

    def forward(self, data):
        prediction, aux_logits = self.model(data)

        return prediction, aux_logits


class Freezed_Resnet(Resnet, torch.nn.Module):
    def __init__(self, load_path, num_classes):
        Resnet.__init__(embeddings=True)
        self.load_model(Resnet, load_path)
        self.fc = torch.nn.Linear(512, num_classes)

    def forward(self, data):
        feature = Resnet.forward(data)
        prediction = self.fc(feature)

        return prediction

    def load_model(Resnet, load_path):
        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))
            checkpoint = torch.load(load_path, map_location='cpu')
            Resnet.load_state_dict(checkpoint['model_state_dict'])
            print("=> loaded checkpoint {}".format(load_path))
        else:
            print("=> no checkpoint found at {}".format(load_path))

        for param in Resnet.parameters():
            param.requires_grad = False


class LSTM(torch.nn.Module):

    def __init__(self, num_classes, lstm_dropout, fc_dropout, embeddings=False, num_layers=2, input_size=512,
                 hidden_size=256):
        super().__init__()

        self.embeddings = embeddings
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                  batch_first=True, dropout=lstm_dropout)
        if not embeddings:
            self.fc = torch.nn.Linear(hidden_size, num_classes)
            self.drop = torch.nn.Dropout(p=fc_dropout)

    def forward(self, data):
        output, (hn, _) = self.lstm(data)  # --> hn shape (layers x batch x hidden)
        last_hidden = hn[-1]  # -->(batch, hidden)

        if not self.embeddings:
            prediction = self.fc(self.drop(last_hidden)).squeeze()
        else:
            prediction = last_hidden.squeeze()

        return prediction, output

    def export_predictions(self, unpacked_sequence, unpacked_sequence_lenghts):
        """

        Args:
            unpacked_sequence: BATCH x MAX_SEQ_LEN x LSTM_HIDDEN_SIZE(32) example 47x50x32
            unpacked_sequence_lenghts: the 'actual' sequence lenghts

        Returns:

            all the predictions after passing through the FC and argmaxed

        """

        all_predictions = []

        for index in range(unpacked_sequence_lenghts.shape[0]):
            seq_len = unpacked_sequence_lenghts[index].item()
            classes = self.fc(unpacked_sequence[index, 0:seq_len, :])
            predictions = torch.argmax(classes, 1)
            all_predictions.append(predictions.cpu().tolist())

        return all_predictions


class GRU(torch.nn.Module):

    def __init__(self, num_classes, lstm_dropout, fc_dropout, embeddings=False, num_layers=2, input_size=512,
                 hidden_size=256):
        super().__init__()

        self.embeddings = embeddings
        self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                batch_first=True, dropout=lstm_dropout)
        if not embeddings:
            self.fc = torch.nn.Linear(hidden_size, num_classes)
            self.drop = torch.nn.Dropout(p=fc_dropout)

    def forward(self, data):
        output, hn = self.gru(data)  # --> hn shape (layers x batch x hidden)
        last_hidden = hn[-1]  # -->(batch, hidden)

        if not self.embeddings:
            prediction = self.fc(self.drop(last_hidden)).squeeze()
        else:
            prediction = last_hidden.squeeze()

        return prediction, output
