import matplotlib.pyplot as plt
from typing import Dict, Tuple

def plot_history(
    result : Dict, 
    figsize : Tuple = (12, 6), 
    accuracy : bool = True, 
    loss : bool = False,
) -> None:
    plt.figure(figsize=figsize)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    for model, runs in result.items():
        for run in runs:
            epochs_range = range(1, len(run['train_acc']) + 1)
            train_acc = run['train_acc']
            val_acc = run['val_acc']
            train_loss = run['train_loss']
            val_loss = run['val_loss']
            
            label_base = f'{model}_{run["optimizer"]}_{run["epoch"]}ep'
            
            if accuracy:
                ax1.plot(epochs_range, train_acc, label=f'{label_base}_train', linewidth=2)
                ax2.plot(epochs_range, val_acc, label=f'{label_base}_val', linewidth=2)
            
            if loss:
                ax1.plot(epochs_range, train_loss, label=f'{label_base}_train_loss', linewidth=2)
                ax2.plot(epochs_range, val_loss, label=f'{label_base}_val_loss', linewidth=2)

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Accuracy' if accuracy else 'Training Loss')
    ax1.set_title('Training Accuracy' if accuracy else 'Training Loss')
    ax1.legend()

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Validation Accuracy' if accuracy else 'Validation Loss')
    ax2.set_title('Validation Accuracy' if accuracy else 'Validation Loss')
    ax2.legend()

    plt.tight_layout()
    plt.show()
