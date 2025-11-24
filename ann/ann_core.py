"""
Artificial Neural Network (ANN) Implementation from Scratch
A simple feedforward neural network with backpropagation
"""

import numpy as np
from typing import List, Tuple, Callable


class ANN:
    """
    A simple Artificial Neural Network with configurable architecture.
    
    Attributes:
        layer_sizes: List of integers representing neurons in each layer
        weights: List of weight matrices for each layer
        biases: List of bias vectors for each layer
        activations: List of activation outputs for each layer
        z_values: List of pre-activation values for each layer
    """
    
    def __init__(self, layer_sizes: List[int], activation: str = 'sigmoid', learning_rate: float = 0.01):
        """
        Initialize the ANN with random weights and biases.
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            activation: Activation function ('sigmoid', 'tanh', 'relu')
            learning_rate: Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.activation_name = activation
        
        # Initialize weights and biases with He initialization
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # He initialization for better gradient flow
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # Storage for forward pass
        self.activations = []
        self.z_values = []
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
    
    def _activation(self, z: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation_name == 'tanh':
            return np.tanh(z)
        elif self.activation_name == 'relu':
            return np.maximum(0, z)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    def _activation_derivative(self, z: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function."""
        if self.activation_name == 'sigmoid':
            a = self._activation(z)
            return a * (1 - a)
        elif self.activation_name == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation_name == 'relu':
            return (z > 0).astype(float)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the network.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            Output of the network of shape (batch_size, output_size)
        """
        self.activations = [X]
        self.z_values = []
        
        current_activation = X
        
        # Forward pass through all layers
        for i in range(self.num_layers - 1):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Use sigmoid for output layer in binary classification
            if i == self.num_layers - 2 and self.layer_sizes[-1] == 1:
                current_activation = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            else:
                current_activation = self._activation(z)
            
            self.activations.append(current_activation)
        
        return current_activation
    
    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Backward propagation to compute gradients and update weights.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            y: Target labels of shape (batch_size, output_size)
        """
        m = X.shape[0]  # batch size
        
        # Compute output layer error
        delta = self.activations[-1] - y
        
        # Backpropagate through layers
        for i in range(self.num_layers - 2, -1, -1):
            # Compute gradients
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
            
            # Propagate error to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._activation_derivative(self.z_values[i - 1])
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Loss value
        """
        m = y_true.shape[0]
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute classification accuracy.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy value
        """
        predictions = (y_pred > 0.5).astype(int)
        accuracy = np.mean(predictions == y_true)
        return accuracy
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              batch_size: int = 32, verbose: bool = True) -> Tuple[List[float], List[float]]:
        """
        Train the neural network using mini-batch gradient descent.
        
        Args:
            X: Training data of shape (num_samples, input_size)
            y: Training labels of shape (num_samples, output_size)
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            verbose: Whether to print training progress
            
        Returns:
            Tuple of (loss_history, accuracy_history)
        """
        n_samples = X.shape[0]
        self.loss_history = []
        self.accuracy_history = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward and backward pass
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch)
            
            # Compute metrics on full dataset
            y_pred_full = self.forward(X)
            loss = self.compute_loss(y, y_pred_full)
            accuracy = self.compute_accuracy(y, y_pred_full)
            
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)
            
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
        
        return self.loss_history, self.accuracy_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data of shape (num_samples, input_size)
            
        Returns:
            Predictions of shape (num_samples, output_size)
        """
        return self.forward(X)
    
    def get_decision_boundary(self, X: np.ndarray, resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate decision boundary for 2D input data.
        
        Args:
            X: Input data to determine boundary range
            resolution: Grid resolution
            
        Returns:
            Tuple of (xx, yy, Z) for plotting
        """
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid).reshape(xx.shape)
        
        return xx, yy, Z
