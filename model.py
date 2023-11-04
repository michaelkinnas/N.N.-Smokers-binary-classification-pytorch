import torch

class SmokersBinaryClassification(torch.nn.Module):
    def __init__(self, input_features, output_features=1):
        super(SmokersBinaryClassification, self).__init__()  
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_features, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=64),
            torch.nn.ReLU(),
            # nn.Linear(in_features=64, out_features=64),
            # nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=output_features)            
        )

    def forward(self, x):
        return self.stack(x)