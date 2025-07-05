import torch
import torch.nn as nn
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(2, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.main(x)

def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator = Generator()
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.to(device)
    generator.eval()
    return generator

def predict_flux(generator, enrichment):
    enrichment = float(enrichment)
    noise = torch.tensor(np.random.uniform(-1., 1., (1, 1)), dtype=torch.float32, device=device)
    input_tensor = torch.tensor([[enrichment, noise.item()]], dtype=torch.float32, device=device)
    output = generator(input_tensor).cpu().detach().numpy()
    flux_value = output[0][1]  # assuming flux is 2nd output
    return round(flux_value, 6)
