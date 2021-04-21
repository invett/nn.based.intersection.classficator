import os

import torch
from torchvision import models


class Resnet18(torch.nn.Module):

    def __init__(self, pretrained=True, embeddings=False, num_classes=None):
        super().__init__()
        self.embeddings = embeddings

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
            return embedding.squeeze()
        else:
            prediction = self.fc(embedding)
            return prediction


class Vgg11(torch.nn.Module):

    def __init__(self, pretrained=True, embeddings=False, num_classes=None):
        super().__init__()
        model = models.vgg11(pretrained=pretrained)
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


class Freezed_Resnet(Resnet18, torch.nn.Module):
    def __init__(self, load_path, num_classes):
        Resnet18.__init__(embeddings=True)
        self.load_model(Resnet18, load_path)
        self.fc = torch.nn.Linear(512, num_classes)

    def forward(self, data):
        feature = Resnet18.forward(data)
        prediction = self.fc(feature)

        return prediction

    def load_model(Resnet18, load_path):
        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))
            checkpoint = torch.load(load_path, map_location='cpu')
            Resnet18.load_state_dict(checkpoint['model_state_dict'])
            print("=> loaded checkpoint {}".format(load_path))
        else:
            print("=> no checkpoint found at {}".format(load_path))

        for param in Resnet18.parameters():
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
