import torch
import torch.nn as nn

class DeepSetArchitecture(nn.Module):
    def __init__(self, N_dim, config):        
        super(DeepSetArchitecture, self).__init__()
        self.N_dim = N_dim
        self.config = config
        activation_functions = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'LeakyReLU': nn.LeakyReLU}

        layers_phi = []
        layers_phi.append(nn.Linear(N_dim, config['neurons_per_layer_phi']))
        layers_phi.append(self.get_activation_function(config['activation_function']))
        for _ in range(config['num_layers_phi'] - 1):
            layers_phi.append(nn.Linear(config['neurons_per_layer_phi'], config['neurons_per_layer_phi']))
            layers_phi.append(self.get_activation_function(config['activation_function']))      
        layers_phi.append(nn.Linear(config['neurons_per_layer_phi'], config['latent_space_dimension']))
        layers_phi.append(self.get_activation_function(config['activation_function']))

        self.phi = nn.Sequential(*layers_phi)
        layers_rho = []
        layers_rho.append(nn.Linear(config['latent_space_dimension'], config['neurons_per_layer_rho']))
        layers_rho.append(self.get_activation_function(config['activation_function']))
        for _ in range(config['num_layers_rho'] - 1):
            layers_rho.append(nn.Linear(config['neurons_per_layer_rho'], config['neurons_per_layer_rho']))
            layers_rho.append(self.get_activation_function(config['activation_function']))
        layers_rho.append(nn.Linear(config['neurons_per_layer_rho'], 1))
        self.rho = nn.Sequential(*layers_rho)
        self.reset_parameters()

    def get_activation_function(self, name):
        activation_functions = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'LeakyReLU': nn.LeakyReLU}
        if name in ['relu', 'LeakyReLU']:
            return activation_functions[name](inplace=True)
        else:
            return activation_functions[name]()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, inputs: torch.Tensor):
        batch_size, num_customers, num_features = inputs.size()

        #print(f"Initial batch_size: {batch_size}, num_customers: {num_customers}, num_features: {num_features}")

        #padding rows before phi
        padding_indices = [
            torch.where(torch.all(row == -1.0e+04, dim=-1))[0].min().item()
            if torch.any(torch.all(row == -1.0e+04, dim=-1))
            else num_customers
            for row in inputs]

        #print(f"Padding indices: {padding_indices}")

        inputs = inputs.view(-1, num_features)  # flattening the cust and feature dim
        #print(f"Flattened inputs shape: {inputs.shape}")

        x = self.phi(inputs)
        #print(f"After phi transformation, x shape: {x.shape}")

        # reshape to [batch_size, num_customers, latent_space_dimension]
        x = x.view(batch_size, num_customers, -1)
        #print(f"Reshaped x to: {x.shape}")

        summed_tensors = []

        for i in range(batch_size):
            first_dummy_index = padding_indices[i]
            summed_tensor = x[i, :first_dummy_index, :].sum(dim=0)
            summed_tensors.append(summed_tensor)
            #print(f"Summed tensor for batch {i}: {summed_tensor}")

        summed_tensor_stack = torch.stack(summed_tensors)
        #print(f"Stacked summed tensors: {summed_tensor_stack.shape}")

        x = self.rho(summed_tensor_stack)
        #print(f"After rho transformation, x shape: {x.shape}")

        cost = x[:, 0]
        #print(f"Cost: {cost}, Num Routes: {num_routes}")

        return cost


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'Phi=' + str(self.phi) \
            + '\n Rho=' + str(self.rho) + ')'