import os
import torch
from typing import Union, Dict, List
from torch.nn import Module
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report


def evaluate(
    model_paths: List[str],
    architectures: Union[Dict[str, Module], Module],
    test_loader: DataLoader,
    device: str = 'cuda'
) -> None:
    """
    Evaluate one or multiple model checkpoints on a test dataset.

    Parameters:
    ----------
    model_paths : List[str]
        List of file paths to saved model checkpoints.
    
    architectures : Union[Dict[str, Module], Module]
        Either a dictionary mapping model name prefixes to model instances,
        or a single model instance to use for all checkpoints.
    
    test_loader : DataLoader
        PyTorch DataLoader for test dataset.
    
    device : str, optional
        Device to run inference on. Default is 'cuda'.
    
    Returns:
    -------
    None
        Prints classification reports to stdout.
    """
    for path in model_paths:
        model_name = os.path.basename(path).split('_')[0]

        if isinstance(architectures, dict):
            model = architectures.get(model_name)
            if model is None:
                print(f"[WARNING] Skipping unknown model architecture: '{model_name}'")
                continue
        else:
            model = architectures

        # Load model checkpoint
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()

        preds, labels = [], []

        # Run inference
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                predicted = outputs.argmax(dim=1)
                preds.extend(predicted.cpu().numpy())
                labels.extend(targets.cpu().numpy())

        print(f"[INFO] Evaluation complete for model: {os.path.basename(path)}")
        print("=" * 60)
        print(classification_report(labels, preds))
        print("=" * 60)
