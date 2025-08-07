import torch
from torch import nn
from typing import Tuple


def generate_all_classes(
    generator: nn.Module,
    n_per_class: int = 999,
    latent_dim: int = 100,
    device: torch.device = torch.device('cuda')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data for all class labels using a conditional generator.

    Args:
        generator (nn.Module): A conditional GAN generator model accepting (latent_vector, class_labels).
        n_per_class (int): Number of samples to generate per class. Default is 999.
        latent_dim (int): Dimensionality of the latent vector z. Default is 100.
        device (torch.device): Device to run the generator on. Default is CUDA.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - generated_data (Tensor): Generated images of shape (10 * n_per_class, ...)
            - labels (Tensor): Corresponding class labels of shape (10 * n_per_class,)
    """
    generator.eval()
    all_generated = []
    all_labels = []

    with torch.no_grad():
        for class_id in range(10):
            z = torch.randn(n_per_class, latent_dim, device=device)
            labels = torch.full((n_per_class,), class_id, dtype=torch.long, device=device)
            generated = generator(z, labels)

            all_generated.append(generated)
            all_labels.append(labels)

    generated_data = torch.cat(all_generated, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return generated_data, labels
