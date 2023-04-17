import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(
            self,
            input_layer_size: int,
            output_layer_size: int,
            hidden_layer_count: int,
            hidden_layer_size: int,
            activation=torch.nn.ReLU,
    ):
        """
        Simple dense neural network with configurable layer count and sizes.
        """
        super().__init__()

        layers = [torch.nn.Linear(input_layer_size, hidden_layer_size), activation()]

        for _ in range(hidden_layer_count):
            layers.append(torch.nn.Linear(hidden_layer_size, hidden_layer_size))
            layers.append(activation())
        layers.append(torch.nn.Linear(hidden_layer_size, output_layer_size))
        self.sequential = torch.nn.Sequential(*layers)

    def reset_parameters(self):
        """ Reset the parameters of the neural network. """
        # go through all layers of the nn...
        for child in self.sequential.children():
            # ... and reset the parameters of all linear layers
            if isinstance(child, torch.nn.Linear):
                child.reset_parameters()

    def forward(self, x):
        """Apply all layers sequentially."""
        return self.sequential(x)

