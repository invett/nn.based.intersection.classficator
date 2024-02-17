import torch


class Resnet(torch.nn.Module):

    def __init__(self, pretrained=True, embeddings_size=512, num_classes=None, version='resnet18'):
        super().__init__()

        self.embedding_size = embeddings_size

        # Crear el modelo base
        if pretrained:
            model = torch.hub.load("pytorch/vision", version, weights="DEFAULT")
        else:
            model = torch.hub.load("pytorch/vision", version)

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

        if embeddings_size is not None:
            self.fc = torch.nn.Linear(model.fc.in_features, embeddings_size)
        elif num_classes is not None:
            self.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            self.softmax = torch.nn.LogSoftmax(dim=1)
        else:
            raise ValueError('embeddings_size and num_classes cannot be both None')

    def forward(self, data):
        x = self.conv1(data)
        x = self.relu(self.bn1(x))
        x = self.maxpool(x)
        feature1 = self.layer1(x)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)
        embedding = self.avgpool(feature4).squeeze()
        if self.embedding_size is not None:
            return embedding
        else:
            prediction = self.softmax(self.fc(embedding))
            return prediction

