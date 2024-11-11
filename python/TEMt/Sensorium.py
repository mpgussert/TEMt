import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import CenterCrop

from einops import rearrange
from .Modules.Resnet18VAE import ResNet18Dec
from torchvision.models import resnet18, ResNet18_Weights

class Resnet18Sensorium(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.code_dim = z_dim
        full_model = resnet18(weights = ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(full_model.children())[:-1])
        self.encfc = nn.Linear(512, 2*self.code_dim)
        self.mu = nn.Linear(2*self.code_dim, self.code_dim)
        self.logvar = nn.Linear(2*self.code_dim, self.code_dim)
        self.mu_lnorm = nn.LayerNorm(self.code_dim)
        self.logvar_lnorm = nn.LayerNorm(self.code_dim)
        self.decoder = ResNet18Dec(z_dim=self.code_dim)

        self.is_frozen = False

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.is_frozen = True
    
    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        self.is_frozen = False

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.encfc(x)
        #means = self.mu(x)
        #logvars = self.logvar(x)
        means = torch.tanh(self.mu_lnorm(self.mu(x))) #center and move things onto [-1,1]
        logvars = torch.tanh(self.logvar_lnorm(self.logvar(x)))

        return means, logvars
    
    def decode(self, code):
        x = self.decoder(code)
        return x

    @staticmethod
    def sample(mean, logvar):
        deviation = torch.exp(logvar*0.5) # in log-space, square root is divide by two
        epsilon = torch.randn_like(deviation)
        sample = mean + deviation*epsilon # + self.code_bias
        return torch.clamp(sample, 0, 1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        code = self.sample(mu, logvar)
        result = self.decode(code)
        return result, code, logvar