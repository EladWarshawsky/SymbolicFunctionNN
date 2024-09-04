import torch
import torch.nn as nn

class SymbolicLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activations = [torch.sin,torch.cos, torch.exp,lambda x: x,lambda x: x**2,lambda x: x**3]):
        """
        input_dim: Number of input features for the layer.
        output_dim: Number of output features for the layer.
        activations: List of activation functions to choose from. eg. sin,cos,exp,identity,x**2,x**3, etc.
        
        support gpu/cpu by default
        """
        super(SymbolicLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activations = activations
        
        # Initialize the parameter array for the layer
        self.layer_params = nn.Parameter(torch.randn(input_dim, output_dim, 5))
    
    def forward(self, x):
        a = self.layer_params[..., 0]  # Shape: [input_dim, output_dim]
        b = self.layer_params[..., 1]  # Shape: [input_dim, output_dim]
        c = self.layer_params[..., 2]  # Shape: [input_dim, output_dim]
        d = self.layer_params[..., 3]  # Shape: [input_dim, output_dim]
        activation_idx = self.layer_params[..., 4].round().long().clamp(0, len(self.activations) - 1)
        
        # Apply transformations in a batched manner
        x_expanded = x.unsqueeze(-1)  # Shape: [batch_size, input_dim, 1]
        a = a.unsqueeze(0)  # Shape: [1, input_dim, output_dim]
        b = b.unsqueeze(0)  # Shape: [1, input_dim, output_dim]
        c = c.unsqueeze(0)  # Shape: [1, input_dim, output_dim]
        d = d.unsqueeze(0)  # Shape: [1, input_dim, output_dim]
        
        # Linear transformation: a * x + b
        linear_transformation = a * x_expanded + b  # Shape: [batch_size, input_dim, output_dim]
        
        # Apply activation functions based on indices
        activations = torch.zeros_like(linear_transformation)
        for idx in range(len(self.activations)):
            mask = (activation_idx == idx)
            activations += self.activations[idx](linear_transformation) * mask.unsqueeze(0).float()
        
        # Apply c and d: c * activation + d
        transformed = c * activations + d  # Shape: [batch_size, input_dim, output_dim]
        
        # Sum over the input dimension to get the final output
        output = transformed.sum(dim=1)  # Shape: [batch_size, output_dim]
        
        return output