import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import CenterCrop

from einops import rearrange
from .Modules import resnet50_encoder as enc
from .Modules import resnet50_decoder as dec
from .Modules.Resnet18VAE import ResNet18Dec
from torchvision.models import resnet18, ResNet18_Weights

def conv_block(h_dims=None, input_chans=3):
        modules = []
        # input_chans = self.input_shape[0]
        if h_dims is None:
            h_dims = [32, 64, 128, 256, 512]

        for i, h in enumerate(h_dims):
            print(i, input_chans, h)
            layer = nn.Sequential(nn.Conv2d(input_chans, h, kernel_size=3, stride=2, padding=1),
                                  nn.ReLU())
            modules.append(layer)
            input_chans = h
        return nn.Sequential(*modules)

def tconv_block(h_dims=None, output_chans=3):
    modules = []
    if h_dims is None:
        h_dims = [32, 64, 128, 256, 512]
        h_dims.reverse()

    o_chans = h_dims[1]
    for i, h in enumerate(h_dims):
        print(i, h, o_chans)
        layer = nn.Sequential(nn.ConvTranspose2d(h, o_chans, kernel_size=3, stride=2, dilation=1),
                                nn.ReLU())
        modules.append(layer)
        o_chans = output_chans if i >= len(h_dims)-2 else h_dims[i+2]
    return nn.Sequential(*modules)

class ResnetSensorium(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = enc.ResNet(enc.Bottleneck, [3,4,6,3], return_indices = True)
        self.input_shape = [3,221,221] #this should be 224 but i get an error if with the implementation i found if so... worrysome bug...
        self.code_dim = 2048

        #post resnet dims = [batch, 2048,1,1]
        with torch.no_grad():
            temp = torch.randn(*self.input_shape).unsqueeze(0)
            out, _ = self.encoder(temp)
        self._transition_shape = out.shape[1:]
        self._transition_size = out.flatten().shape[0] #should be 2048

        self.ehlayer = torch.nn.Linear(self._transition_size, 2*self.code_dim)
        self.mu = nn.Linear(2*self.code_dim, self.code_dim)
        self.logvar = nn.Linear(2*self.code_dim, self.code_dim)
        #self.code_bias = nn.Parameter(torch.randn(1, self.code_dim)*0.1)
        self.dcomp = nn.Linear(self.code_dim, 2*self.code_dim)
        self.dhlayer = torch.nn.Linear(2*self.code_dim, self._transition_size)

        self.decoder = dec.ResNet(dec.Bottleneck, [3,6,4,3])

    def encode(self, x):
        x, x_idx = self.encoder(x)
        x = x.view(-1,self._transition_size)
        x = torch.relu(self.ehlayer(x))
        means = torch.sigmoid(self.mu(x))
        logvars = torch.tanh(self.logvar(x))

        return means, logvars, x_idx

    def sample(self, means, logvars):
        deviation = torch.exp(logvars*0.5)
        epsilon = torch.randn_like(deviation)
        sample = means + deviation*epsilon # + self.code_bias
        return torch.clamp(sample, 0, 1)

    def decode(self, codes, x_idx):
        x = self.dcomp(codes)
        x = torch.relu(self.dhlayer(x))
        x = x.view(-1, *self._transition_shape)
        x = self.decoder(x, x_idx)
        x = torch.clamp(x,0,1)
        return x

    def forward(self, x):
        means, logvars, x_idx = self.encode(x)
        sample_codes = self.sample(means, logvars)
        samples = self.decode(sample_codes, x_idx)
        return samples, means, logvars

class OpticalEncoder(nn.Module):
    def __init__(self, chans: int, rows: int, cols: int, encoding_dim: int, h_dims: list = None):
        super().__init__()

        self.input_shape = [chans, rows, cols]
        self.code_dim = encoding_dim
        self._image_batch_norm = nn.BatchNorm2d(chans)
        # self._code_batch_norm = nn.BatchNorm1d(self.code_dim)

        self.encoder = conv_block(h_dims, input_chans=chans)
        with torch.no_grad():
            temp = torch.randn(*self.input_shape).unsqueeze(0)
            out = self.encoder(temp)
        self._transition_shape = out.shape[1:]
        self._transition_size = out.flatten().shape[0]

        print("transition shape ", self._transition_shape, self._transition_size)
        print("pre embedding compression ratio ", self._transition_size/(torch.prod(torch.tensor(self.input_shape))))

        self.ehlayer = torch.nn.Linear(self._transition_size, 2*self.code_dim)
        self.mu = nn.Linear(2*self.code_dim, self.code_dim)
        self.logvar = nn.Linear(2*self.code_dim, self.code_dim)
        self.code_bias = nn.Parameter(torch.randn(1, self.code_dim)*0.1)
        self.dcomp = nn.Linear(self.code_dim, 2*self.code_dim)
        self.dhlayer = torch.nn.Linear(2*self.code_dim, self._transition_size)

        self.mu_lnorm = nn.LayerNorm(self.code_dim)
        self.logvar_lnorm = nn.LayerNorm(self.code_dim)

        if h_dims is not None:
            h_dims.reverse()
        self.decoder = tconv_block(h_dims, output_chans=chans)
        self.finalize = CenterCrop(self.input_shape[1]) #nn.ConvTranspose2d(chans, chans, kernel_size=3, dilation=1)

    def encode(self, images):
        x = self._image_batch_norm(images)
        x = self.encoder(x)
        x = x.view(-1,self._transition_size)
        x = torch.relu(self.ehlayer(x))
        means = torch.tanh(self.mu_lnorm(self.mu(x))) #center and move things onto [-1,1]
        logvars = torch.tanh(self.logvar_lnorm(self.logvar(x)))

        return means, logvars

    def sample(self, means, logvars):
        deviation = torch.exp(logvars*0.5)
        epsilon = torch.randn_like(deviation)
        
        return means + deviation*epsilon + self.code_bias

    def decode(self, codes, bias_included = True):
        if not bias_included:
            codes = codes + self.code_bias
        x = self.dcomp(codes)
        x = torch.relu(self.dhlayer(x))
        x = x.view(-1, *self._transition_shape)
        x = self.decoder(x)
        x = self.finalize(x)
        images = torch.sigmoid(x)

        return images

    def forward(self, images):
        codes, logvars = self.encode(images)
        sample_codes = self.sample(codes, logvars)
        samples = self.decode(sample_codes)

        return samples, codes, logvars

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
        deviation = torch.exp(logvar*0.5) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(deviation)
        sample = mean + deviation*epsilon # + self.code_bias
        return torch.clamp(sample, 0, 1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        code = self.sample(mu, logvar)
        result = self.decode(code)
        return result, code, logvar