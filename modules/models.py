#!/usr/bin/env python

import torch

class GAIARegressor(torch.nn.Module):
    def __init__(self, input_dim: int = 5, output_dim: int = 3) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        sizes = [input_dim, 16, 32, 64, 128, 256, 512, output_dim]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(torch.nn.ReLU())
        
        self.fcl = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @x: torch.Tensor((B, 5))
        @return: torch.Tensor((B, 3))
        """
        assert x.shape[1] == self.input_dim, 'Input tensor does not match required dimensions'

        return self.fcl(x)


class SDSSRegressor(torch.nn.Module):
    def __init__(self, input_dim: int = 3, output_dim: int = 5) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        sizes = [input_dim, 16, 32, 64, 128, 256, 512, output_dim]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(torch.nn.ReLU())
        
        self.fcl = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @x: torch.Tensor((B, 3))
        @return: torch.Tensor((B, 5))
        """
        assert x.shape[1] == self.input_dim, 'Input tensor does not match required dimensions'

        return self.fcl(x)

model_factory : dict[str, torch.nn.Module] = {
    'gaia': GAIARegressor,
    'sdss': SDSSRegressor
}

def save_model(model: torch.nn.Module) -> None:
    from torch import save
    from os import path
    
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError('model type \'%s\' not supported' % str(type(model)))


def load_model(model: str) -> torch.nn.Module:
    from torch import load
    from os import path
    r: torch.nn.Module = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r