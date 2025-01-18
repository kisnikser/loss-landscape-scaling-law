import torch
import torch.nn as nn

class MLP(nn.Module):
    r"""
    Multi-Layer Perceptron (MLP) model.

    Args:
        sizes_list list(int): list of layers sizes
        activation_class: activation after all linear layers
    """
    def __init__(self, sizes_list, activation_class = nn.ReLU):
        super(MLP, self).__init__()
        self.layers = []
        for in_size, out_size in zip(sizes_list[:-2], sizes_list[1:-1]):
            self.layers.append(nn.Sequential(
                nn.Linear(in_size, out_size),
                activation_class()
            ))
        self.layers.append(nn.Linear(sizes_list[-2], sizes_list[-1]))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.layers(x)
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
