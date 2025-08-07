import os
from glob import glob
from pathlib import Path
from typing import Tuple
from torchvision import transforms
from torchvision.datasets import ImageFolder

def Image_Processing(root: str, image_size: Tuple[int, int] = (180, 180)) -> ImageFolder:
    """
    Loads an image dataset from a given directory using torchvision's ImageFolder.

    Args:
        root (str): Root directory containing class-labeled image folders.
        image_size (Tuple[int, int], optional): Desired image size (height, width). Defaults to (180, 180).

    Returns:
        torchvision.datasets.ImageFolder: Dataset with images transformed and labeled by folder structure.
    """
    image_dir = Path(root)

    #Raise an Error if the folder doesn't exist
    if not image_dir.exists() or not any(image_dir.iterdir()):
        raise FileNotFoundError(f"[load_image_dataset] Directory '{root}' does not exist or is empty.")
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(root=str(image_dir), transform=transform)
    return dataset