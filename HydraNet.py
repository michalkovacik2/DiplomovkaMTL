from torch import nn
from torchvision import models
from collections import OrderedDict


class HydraNet(nn.Module):
    def __init__(self, backbone, head_age=True, head_gender=True, head_ethnicity=True):
        super().__init__()
        if backbone == 'resnet18':
            self.net = models.resnet18(pretrained=True)
        elif backbone == 'resnet101':
            self.net = models.resnet101(pretrained=True)
        else:
            raise Exception(f'Unrecognized backbone: {backbone}')

        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Sequential()

        self.net.fc1 = None
        self.net.fc2 = None
        self.net.fc3 = None

        if head_age:
            self.net.fc1 = nn.Sequential(OrderedDict(
                [('linear1', nn.Linear(self.n_features, self.n_features)),
                 ('relu1', nn.ReLU()),
                 ('final1', nn.Linear(self.n_features, 1))]))

        if head_gender:
            self.net.fc2 = nn.Sequential(OrderedDict(
                [('linear2', nn.Linear(self.n_features, self.n_features)),
                 ('relu2', nn.ReLU()),
                 ('final2', nn.Linear(self.n_features, 1))]))

        if head_ethnicity:
            self.net.fc3 = nn.Sequential(OrderedDict(
                [('linear3', nn.Linear(self.n_features, self.n_features)),
                 ('relu3', nn.ReLU()),
                 ('final3', nn.Linear(self.n_features, 5))]))

    def forward(self, x):
        age_head = None if self.net.fc1 is None else self.net.fc1(self.net(x))
        gender_head = None if self.net.fc2 is None else self.net.fc2(self.net(x))
        ethnicity_head = None if self.net.fc3 is None else self.net.fc3(self.net(x))
        return age_head, gender_head, ethnicity_head
