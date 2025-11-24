"""
Visualization Utilities for ANN Demo
Provides plotting functions for decision boundaries, training curves, and network architecture
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Gradio
from typing import Tuple, List, Optional
import io
from PIL import Image


def plot_decision_boundary(X: np.ndarray, y: np.ndarray, model, title: str = "Decision Boundary") -> Image.Image:
    """
    Plot decision boundary for a 2D classification problem.
    
    Args:
        X: Input features of shape (n_samples, 2)
        y: Labels of shape (n_samples, 1)
        model: Trained ANN model with get_decision_boundary method
        title: Plot title
        
    Returns:
        PIL Image object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get decision boundary
    xx, yy, Z = model.get_decision_boundary(X, resolution=200)
    
    # Plot decision boundary
    contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Plot data points
    y_flat = y.flatten()
    scatter = ax.scatter(X[y_flat == 0, 0], X[y_flat == 0, 1], 
                        c='blue', marker='o', s=100, edgecolors='black', 
                        linewidths=1.5, label='Class 0', alpha=0.8)
    ax.scatter(X[y_flat == 1, 0], X[y_flat == 1, 1], 
              c='red', marker='s', s=100, edgecolors='black', 
              linewidths=1.5, label='Class 1', alpha=0.8)
    
    # Styling
    ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Prediction Probability', fontsize=11)
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img


def plot_training_curves(loss_history: List[float], accuracy_history: List[float]) -> Image.Image:
    """
    Plot training loss and accuracy curves.
    
    Args:
        loss_history: List of loss values over epochs
        accuracy_history: List of accuracy values over epochs
        
    Returns:
        PIL Image object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(loss_history) + 1)
    
    # Plot loss
    ax1.plot(epochs, loss_history, 'b-', linewidth=2.5, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    
    # Plot accuracy
    ax2.plot(epochs, accuracy_history, 'g-', linewidth=2.5, label='Training Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Training Accuracy Over Time', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img


def plot_network_architecture(layer_sizes: List[int]) -> Image.Image:
    """
    Visualize the neural network architecture.
    
    Args:
        layer_sizes: List of layer sizes [input, hidden1, ..., output]
        
    Returns:
        PIL Image object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    num_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)
    
    # Layer positions
    layer_x = np.linspace(0, 10, num_layers)
    
    # Draw neurons
    for i, (x, size) in enumerate(zip(layer_x, layer_sizes)):
        # Center neurons vertically
        y_positions = np.linspace(0, max_neurons - 1, size)
        y_offset = (max_neurons - size) / 2
        y_positions += y_offset
        
        # Draw neurons
        for y in y_positions:
            circle = plt.Circle((x, y), 0.3, color='steelblue', ec='black', linewidth=2, zorder=3)
            ax.add_patch(circle)
        
        # Add layer labels
        if i == 0:
            label = f'Input\n({size})'
        elif i == num_layers - 1:
            label = f'Output\n({size})'
        else:
            label = f'Hidden {i}\n({size})'
        
        ax.text(x, -1.5, label, ha='center', va='top', fontsize=11, fontweight='bold')
    
    # Draw connections (sample, not all to avoid clutter)
    for i in range(num_layers - 1):
        x1, x2 = layer_x[i], layer_x[i + 1]
        size1, size2 = layer_sizes[i], layer_sizes[i + 1]
        
        y1_positions = np.linspace(0, max_neurons - 1, size1)
        y1_offset = (max_neurons - size1) / 2
        y1_positions += y1_offset
        
        y2_positions = np.linspace(0, max_neurons - 1, size2)
        y2_offset = (max_neurons - size2) / 2
        y2_positions += y2_offset
        
        # Draw sample connections (not all to avoid clutter)
        sample_rate = max(1, min(size1, size2) // 5)
        for j, y1 in enumerate(y1_positions[::sample_rate]):
            for y2 in y2_positions[::sample_rate]:
                ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=0.5, zorder=1)
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(-3, max_neurons + 1)
    ax.axis('off')
    ax.set_title('Neural Network Architecture', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img


def plot_all_datasets() -> Image.Image:
    """
    Plot all available datasets in a grid.
    
    Returns:
        PIL Image object
    """
    from datasets import DATASETS
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, generator) in enumerate(DATASETS.items()):
        X, y = generator(n_samples=200)
        y_flat = y.flatten()
        
        ax = axes[idx]
        ax.scatter(X[y_flat == 0, 0], X[y_flat == 0, 1], 
                  c='blue', marker='o', s=50, alpha=0.6, label='Class 0')
        ax.scatter(X[y_flat == 1, 0], X[y_flat == 1, 1], 
                  c='red', marker='s', s=50, alpha=0.6, label='Class 1')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Available Datasets', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img


def create_comparison_plot(X: np.ndarray, y: np.ndarray, model, 
                          loss_history: List[float], accuracy_history: List[float]) -> Image.Image:
    """
    Create a comprehensive comparison plot with decision boundary and training curves.
    
    Args:
        X: Input features
        y: Labels
        model: Trained model
        loss_history: Training loss history
        accuracy_history: Training accuracy history
        
    Returns:
        PIL Image object
    """
    fig = plt.figure(figsize=(16, 5))
    
    # Decision boundary
    ax1 = plt.subplot(1, 3, 1)
    xx, yy, Z = model.get_decision_boundary(X, resolution=200)
    contour = ax1.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    y_flat = y.flatten()
    ax1.scatter(X[y_flat == 0, 0], X[y_flat == 0, 1], 
               c='blue', marker='o', s=80, edgecolors='black', 
               linewidths=1.5, label='Class 0', alpha=0.8)
    ax1.scatter(X[y_flat == 1, 0], X[y_flat == 1, 1], 
               c='red', marker='s', s=80, edgecolors='black', 
               linewidths=1.5, label='Class 1', alpha=0.8)
    
    ax1.set_xlabel('Feature 1', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Feature 2', fontsize=11, fontweight='bold')
    ax1.set_title('Decision Boundary', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Loss curve
    ax2 = plt.subplot(1, 3, 2)
    epochs = range(1, len(loss_history) + 1)
    ax2.plot(epochs, loss_history, 'b-', linewidth=2.5)
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(epochs, accuracy_history, 'g-', linewidth=2.5)
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax3.set_title('Training Accuracy', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img
