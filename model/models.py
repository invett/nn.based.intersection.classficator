import os

import torch
from torchvision import models


class Resnet(torch.nn.Module):

    def __init__(self, pretrained=True, embeddings=False, num_classes=None, version='resnet18', logits=False):
        super().__init__()
        self.embeddings = embeddings
        self.version = version
        self.logits = logits

        if version == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        if version == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
        if version == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        if version == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
        if version == 'resnet152':
            model = models.resnet152(pretrained=pretrained)

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
            if version == 'resnet50' or version == 'resnet101' or version == 'resnet152':
                self.reducer = torch.nn.Linear(2048, 512)

        if self.logits:
            self.softmax = torch.nn.LogSoftmax()

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
                embedding = self.reducer(embedding.squeeze())
                return embedding
        else:
            prediction = self.fc(embedding.squeeze())
            if self.logits:
                return self.softmax(prediction)
            else:
                return prediction


class VGG(torch.nn.Module):

    def __init__(self, pretrained=True, embeddings=False, num_classes=None, version='vgg11', logits=False):
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
        self.logits = logits

        self.features = model.features
        self.avgpool = model.avgpool

        if embeddings:
            self.classifier = model.classifier
            self.classifier[6] = torch.nn.Linear(4096, 512)
        else:
            self.classifier = model.classifier
            self.classifier[6] = torch.nn.Linear(4096, num_classes)

        if self.logits:
            self.softmax = torch.nn.LogSoftmax()

    def forward(self, data):
        features = self.features(data)
        avg = self.avgpool(features)
        avg = torch.flatten(avg, start_dim=1)
        prediction = self.classifier(avg)

        if self.logits and not self.embeddings:
            prediction = self.softmax(prediction)

        return prediction


class Mobilenet_v3(torch.nn.Module):

    def __init__(self, pretrained=True, embeddings=False, num_classes=None, version='mobilenet_v3_small', logits=False):
        super().__init__()

        self.embeddings = embeddings
        self.version = version
        self.logits = logits

        if version == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(pretrained=pretrained)
        if version == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(pretrained=pretrained)

        self.features = model.features
        self.avgpool = model.avgpool

        if embeddings:
            self.classifier = model.classifier
            if version == 'mobilenet_v3_small':
                self.classifier[3] = torch.nn.Linear(1024, 512)
            else:
                self.classifier[3] = torch.nn.Linear(1280, 512)
        else:
            self.classifier = model.classifier
            if version == 'mobilenet_v3_small':
                self.classifier[3] = torch.nn.Linear(1024, num_classes)
            else:
                self.classifier[3] = torch.nn.Linear(1280, num_classes)

        if self.logits:
            self.softmax = torch.nn.LogSoftmax()

    def forward(self, data):
        features = self.features(data)
        avg = self.avgpool(features)
        prediction = self.classifier(avg.squeeze())

        if self.logits and not self.embeddings:
            prediction = self.softmax(prediction)

        return prediction


class Inception_v3(torch.nn.Module):

    def __init__(self, pretrained=True, embeddings=False, num_classes=None, logits=False):
        super().__init__()

        self.embeddings = embeddings
        self.logits = logits

        self.model = models.inception_v3(pretrained=pretrained)

        if embeddings:
            self.model.fc = torch.nn.Linear(2048, 512)
            self.model.AuxLogits.fc = torch.nn.Linear(768, 512)
        else:
            self.model.fc = torch.nn.Linear(2048, num_classes)
            self.model.AuxLogits.fc = torch.nn.Linear(768, num_classes)

        if self.logits:
            self.softmax = torch.nn.LogSoftmax()

    def forward(self, data):
        if self.training:
            prediction, aux_logits = self.model(data)

            return prediction, aux_logits
        else:
            prediction = self.model(data)

            if self.logits and not self.embeddings:
                prediction = self.softmax(prediction)

            return prediction


class Classifier(torch.nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.fc = torch.nn.Linear(512, num_classes)
        #self.fc2 = torch.nn.Linear(128, num_classes)
        #self.dropout = torch.nn.Dropout(p=0.2)
        #self.elu = torch.nn.ELU()

    def forward(self, data):
        #x = self.dropout(self.elu(self.fc1(data)))
        prediction = self.fc(data)

        return prediction


class Freezed_Model(torch.nn.Module):
    def __init__(self, model, load_path, num_classes):
        super().__init__()
        self.model = model
        self.load_model(load_path)
        self.classifier = Classifier(num_classes=num_classes)

    def forward(self, data):
        feature = self.model(data)
        if isinstance(feature, tuple):
            prediction = self.classifier(feature[0])
            aux_prediction = self.classifier(feature[1])
            return prediction, aux_prediction
        else:
            prediction = self.classifier(feature)
            return prediction

    def load_model(self, load_path):
        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))
            checkpoint = torch.load(load_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("=> loaded checkpoint {}".format(load_path))
        else:
            print("=> no checkpoint found at {}".format(load_path))

        for param in self.model.parameters():
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
