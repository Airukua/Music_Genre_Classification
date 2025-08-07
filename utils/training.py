import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Tuple, Union, Iterable, Type
from pathlib import Path
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        save_dir: Path,
        early_stopping: bool = False,
        patience: int = 5,
    ):
        """
        Trainer for handling training and validation of multiple models.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            device (torch.device): Torch device to use.
            save_dir (Path): Path to save model checkpoints.
            early_stopping (bool): Enable early stopping.
            patience (int): Patience for early stopping.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.early_stopping = early_stopping
        self.patience = patience

    def train_and_validate(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        num_epochs: int,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, num_epochs + 1):
            # --- Training ---
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            train_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False)

            for data, labels in train_bar:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = outputs.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

                train_bar.set_postfix(
                    loss=train_loss / len(self.train_loader),
                    acc=train_correct / train_total,
                )

            epoch_train_loss = train_loss / len(self.train_loader)
            epoch_train_acc = train_correct / train_total
            history["train_loss"].append(epoch_train_loss)
            history["train_acc"].append(epoch_train_acc)

            # --- Validation ---
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            val_bar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]", leave=False)

            with torch.no_grad():
                for data, labels in val_bar:
                    data, labels = data.to(self.device), labels.to(self.device)
                    outputs = model(data)
                    loss = loss_fn(outputs, labels)

                    val_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

                    val_bar.set_postfix(
                        val_loss=val_loss / len(self.val_loader),
                        acc=val_correct / val_total,
                    )

            epoch_val_loss = val_loss / len(self.val_loader)
            epoch_val_acc = val_correct / val_total
            history["val_loss"].append(epoch_val_loss)
            history["val_acc"].append(epoch_val_acc)

            if epoch % 10 == 0 or epoch == num_epochs:
                print(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
                    f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}"
                )

            # Early Stopping inspired by https://stackoverflow.com/questions/57295273/earlystopping-not-stopping-model-despite-loading-best-weights
            if self.early_stopping:
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        print(f"[EARLY STOP] No improvement in {self.patience} epochs. Stopping at epoch {epoch}.")
                        break

        return (
            history["train_loss"],
            history["train_acc"],
            history["val_loss"],
            history["val_acc"],
        )
    
    #This code inspired by the implementation found in: https://discuss.pytorch.org/t/reset-model-weights/19180/4
    def reset_weights(self, m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    def train_models(
        self,
        models: Dict[str, Callable[[], nn.Module]],
        epochs: Iterable[int],
        optimizers: Iterable[Tuple[Type[optim.Optimizer], Dict[str, float]]] = None,
    ) -> Dict[str, List[Dict[str, Union[int, float]]]]:
        """
        Train and validate multiple models with multiple optimizers and configs.

        Args:
            models (Dict): model_name -> constructor function
            epochs (Iterable): list of epoch counts to try
            optimizers (List): List of (OptimizerClass, config_dict)

        Returns:
            Dict[str, List[Dict]]: Training histories for each model variant
        """
        if optimizers is None:
            optimizers = [
                (optim.Adam, {"lr": 1e-4}),
                (optim.RMSprop, {"lr": 5e-4}),
            ]

        results: Dict[str, List[Dict[str, Union[int, float]]]] = {model_name: [] for model_name in models}

        for model_name, model_fn in models.items():
            for num_epochs in epochs:
                for optimizer_cls, opt_cfg in optimizers:
                    print(f"[INFO] Training: {model_name} | Epochs: {num_epochs} | Optimizer: {optimizer_cls.__name__} | Config: {opt_cfg}")

                    model = model_fn().to(self.device)
                    optimizer = optimizer_cls(model.parameters(), **opt_cfg)
                    loss_fn = nn.CrossEntropyLoss()

                    train_loss, train_acc, val_loss, val_acc = self.train_and_validate(
                        model=model,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        num_epochs=num_epochs,
                    )

                    results[model_name].append({
                        "epoch": len(train_loss),
                        "optimizer": optimizer_cls.__name__,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc
                    })

                    model_path = self.save_dir / f"{model_name}_{optimizer_cls.__name__}_epoch{len(train_loss)}.pt"
                    torch.save({
                        "model_name": model_name,
                        "model_params": model_fn.__name__,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer_cls.__name__,
                        "epochs": len(train_loss)
                    }, model_path)

                    print(f"[INFO] Saved: {model_path}")

                    # Weight Reset
                    model.apply(self.reset_weights)

        return results
