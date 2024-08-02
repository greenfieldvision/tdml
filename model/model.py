import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

    def custom_regularized_parameters(self):
        return []

    def get_custom_regularization_loss(self):
        return 0.0
