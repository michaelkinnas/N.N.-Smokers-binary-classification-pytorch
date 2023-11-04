import torch
from torch import nn
from pathlib import Path


class SmokersBinaryClassification(torch.nn.Module):
    def __init__(self, input_features, output_features=1):
        super(SmokersBinaryClassification, self).__init__()  
        self.stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=output_features)
        )
       
    def forward(self, x):
        return self.stack(x)

MODEL_PATH = Path('saved_models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = 'in-64-64-out-83p35-10000epoch.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


loaded_model = SmokersBinaryClassification(input_features=16)
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer= torch.optim.Adam(loaded_model.parameters(), lr=0.01)

data = torch.tensor([0.0, 40, 170, 79, 118, 75, 91, 174, 40, 100, 134, 15.3, 0.8, 27.1, 26.0, 38], dtype=torch.float32) # Mine

# Single prediction output
with torch.inference_mode():
    y_hat = loaded_model(data)

    print(f'y_hat: {y_hat}')
   
    y_test_pred = torch.round(torch.sigmoid(y_hat))
if y_test_pred == 1: 
    print(f'Is smoker with sigmoid output {torch.sigmoid(y_hat).item():.5f}')
else:
    print(f'Is not smoker with sigmoid output {torch.sigmoid(y_hat).item():.5f}')