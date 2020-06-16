import os

import torch
from torch import nn


class Classifier(nn.Module):

    def __init__(self, feature_len, category_num, checkpoint_path=None):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(feature_len, 1024),
            nn.ReLU(),
            nn.Linear(1024, category_num),
            nn.Sigmoid())

        if checkpoint_path is not None:
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                self.load_state_dict(checkpoint)
                print('Checkpoint loaded from %s' % checkpoint_path)
            else:
                print('Attention: %s not exist' % checkpoint_path)

    def forward(self, features):
        return self.model(features)
