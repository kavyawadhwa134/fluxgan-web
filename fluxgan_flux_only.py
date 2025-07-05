# fluxgan_flux_only.py

import torch
import torch.nn as nn
import numpy as np

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

def sample_noise(num_samples, dim):
    return torch.Tensor(np.random.uniform(-1., 1., size=(num_samples, dim)))

def load_generator_model(path, device):
    model = Generator().to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.eval()
    return model

def predict_flux(enrichment_value, generator, device):
    noise_dim = 2
    tolerance = 0.1
    num_samples = 10000

    noise = sample_noise(num_samples, noise_dim).to(device)
    with torch.no_grad():
        output = generator(noise).cpu().numpy()

    enrichments = output[:, 0]
    fluxes = output[:, 1]

    mask = (enrichments >= enrichment_value - tolerance) & (enrichments <= enrichment_value + tolerance)
    matching_flux = fluxes[mask]

    if len(matching_flux) > 0:
        avg_flux = float(np.mean(matching_flux))
    else:
        avg_flux = None

    return avg_flux
