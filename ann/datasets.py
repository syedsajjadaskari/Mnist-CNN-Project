"""
Dataset Generation and Loading Utilities for ANN Demo
Provides various synthetic and real datasets for testing neural networks
"""

import numpy as np
from typing import Tuple
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split


def generate_xor(n_samples: int = 200, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate XOR dataset - a classic non-linearly separable problem.
    
    Args:
        n_samples: Number of samples to generate
        noise: Amount of noise to add
        
    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    np.random.seed(42)
    
    # Generate four clusters
    n_per_class = n_samples // 4
    
    # Class 0: top-left and bottom-right
    X0_1 = np.random.randn(n_per_class, 2) * noise + np.array([0, 1])
    X0_2 = np.random.randn(n_per_class, 2) * noise + np.array([1, 0])
    X0 = np.vstack([X0_1, X0_2])
    y0 = np.zeros((n_per_class * 2, 1))
    
    # Class 1: top-right and bottom-left
    X1_1 = np.random.randn(n_per_class, 2) * noise + np.array([1, 1])
    X1_2 = np.random.randn(n_per_class, 2) * noise + np.array([0, 0])
    X1 = np.vstack([X1_1, X1_2])
    y1 = np.ones((n_per_class * 2, 1))
    
    # Combine
    X = np.vstack([X0, X1])
    y = np.vstack([y0, y1])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def generate_circles(n_samples: int = 200, noise: float = 0.1, factor: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate concentric circles dataset.
    
    Args:
        n_samples: Number of samples to generate
        noise: Standard deviation of Gaussian noise
        factor: Scale factor between inner and outer circle
        
    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    np.random.seed(42)
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=42)
    y = y.reshape(-1, 1)
    return X, y


def generate_moons(n_samples: int = 200, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two interleaving half circles (moons).
    
    Args:
        n_samples: Number of samples to generate
        noise: Standard deviation of Gaussian noise
        
    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    np.random.seed(42)
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    y = y.reshape(-1, 1)
    return X, y


def generate_spiral(n_samples: int = 200, noise: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate spiral dataset - a challenging non-linear problem.
    
    Args:
        n_samples: Number of samples to generate
        noise: Amount of noise to add
        
    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    np.random.seed(42)
    n_per_class = n_samples // 2
    
    # Generate spiral
    theta = np.sqrt(np.random.rand(n_per_class)) * 2 * np.pi
    
    # Class 0
    r0 = 2 * theta + np.pi
    X0 = np.c_[r0 * np.cos(theta), r0 * np.sin(theta)] + np.random.randn(n_per_class, 2) * noise
    
    # Class 1
    r1 = -2 * theta - np.pi
    X1 = np.c_[r1 * np.cos(theta), r1 * np.sin(theta)] + np.random.randn(n_per_class, 2) * noise
    
    X = np.vstack([X0, X1])
    y = np.vstack([np.zeros((n_per_class, 1)), np.ones((n_per_class, 1))])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def generate_linear(n_samples: int = 200, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate linearly separable dataset.
    
    Args:
        n_samples: Number of samples to generate
        noise: Standard deviation of Gaussian noise
        
    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=2.0,
        random_state=42
    )
    y = y.reshape(-1, 1)
    return X, y


def generate_blobs(n_samples: int = 200, centers: int = 4, cluster_std: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate blob clusters dataset.
    
    Args:
        n_samples: Number of samples to generate
        centers: Number of cluster centers
        cluster_std: Standard deviation of clusters
        
    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    np.random.seed(42)
    from sklearn.datasets import make_blobs
    
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=2,
        cluster_std=cluster_std,
        random_state=42
    )
    
    # Convert to binary classification
    y = (y % 2).reshape(-1, 1)
    return X, y


def normalize_data(X: np.ndarray) -> np.ndarray:
    """
    Normalize features to zero mean and unit variance.
    
    Args:
        X: Input features
        
    Returns:
        Normalized features
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (X - mean) / std


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.
    
    Args:
        X: Features
        y: Labels
        test_size: Proportion of data to use for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)


# Dataset registry for easy access
DATASETS = {
    'XOR': generate_xor,
    'Circles': generate_circles,
    'Moons': generate_moons,
    'Spiral': generate_spiral,
    'Linear': generate_linear,
    'Blobs': generate_blobs
}


def get_dataset(name: str, n_samples: int = 200, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a dataset by name.
    
    Args:
        name: Name of the dataset
        n_samples: Number of samples to generate
        **kwargs: Additional arguments for dataset generation
        
    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    
    return DATASETS[name](n_samples=n_samples, **kwargs)


if __name__ == "__main__":
    # Test all datasets
    print("Testing dataset generation...")
    for name in DATASETS.keys():
        X, y = get_dataset(name)
        print(f"{name:10s} - X shape: {X.shape}, y shape: {y.shape}, "
              f"Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")
