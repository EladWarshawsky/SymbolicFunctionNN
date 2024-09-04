import torch.nn as nn
import torch
from symbolic_layer import SymbolicLayer

class GaborLayer2D(nn.Module):
    '''
    reference: https://github.com/vishwa91/wire
        Implicit representation with Gabor nonlinearity with 2D activation function
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=True):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.float
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = SymbolicLayer(in_features,
                                out_features)
        
        # Second Gaussian window
        self.scale_orth = SymbolicLayer(in_features,
                                    out_features)
    
    def forward(self, input):
        lin = self.linear(input)
        
        scale_x = lin
        scale_y = self.scale_orth(input)
        
        # clampint to aovid nan outputs
        freq_term = torch.exp(torch.clamp(self.omega_0*lin,max = 20))
        
        arg = scale_x.abs().square() + scale_y.abs().square()
        # clamping to avoid nan outputs
        gauss_term = torch.exp(torch.clamp(-self.scale_0*self.scale_0*arg,min=-20.0, max=20.0))
                
        return freq_term*gauss_term
    