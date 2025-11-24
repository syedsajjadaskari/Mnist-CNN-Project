# 🧠 ANN Demo - Interactive Neural Network Trainer

An interactive demonstration of Artificial Neural Networks (ANN) built from scratch using NumPy. This project provides a web-based interface to train neural networks on various datasets and visualize their decision boundaries in real-time.

![ANN Demo](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **🎯 Multiple Datasets**: XOR, Circles, Moons, Spiral, Linear, and Blobs
- **🏗️ Configurable Architecture**: Design your own network with custom hidden layers
- **📊 Real-time Visualization**: See decision boundaries and training progress
- **🎨 Interactive Interface**: Built with Gradio for easy experimentation
- **🔧 Hyperparameter Tuning**: Adjust learning rate, epochs, and activation functions
- **📈 Training Metrics**: Track loss and accuracy over time

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Run the Demo

```bash
# Launch the interactive web interface
python demo_app.py
```

The demo will open in your browser at `http://127.0.0.1:7860`

## 📁 Project Structure

```
ann/
├── ann_core.py       # Core ANN implementation with backpropagation
├── datasets.py       # Dataset generation utilities
├── visualizer.py     # Visualization functions
├── demo_app.py       # Gradio web interface
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## 🎓 How It Works

### Neural Network Architecture

The ANN is implemented from scratch with:

1. **Forward Propagation**: Computes predictions by passing inputs through layers
2. **Backpropagation**: Calculates gradients using the chain rule
3. **Gradient Descent**: Updates weights to minimize loss
4. **Activation Functions**: Sigmoid, Tanh, and ReLU

### Available Datasets

- **XOR**: Classic non-linearly separable problem
- **Circles**: Concentric circles requiring non-linear separation
- **Moons**: Two interleaving half circles
- **Spiral**: Challenging spiral pattern
- **Linear**: Linearly separable data
- **Blobs**: Clustered data points

## 🎮 Usage Examples

### Example 1: XOR Problem

```python
from ann_core import ANN
from datasets import generate_xor

# Generate XOR dataset
X, y = generate_xor(n_samples=200)

# Create and train network
model = ANN([2, 8, 4, 1], activation='tanh', learning_rate=0.1)
model.train(X, y, epochs=500)

# Make predictions
predictions = model.predict(X)
```

### Example 2: Custom Dataset

```python
from datasets import get_dataset
from ann_core import ANN

# Get any dataset
X, y = get_dataset('Circles', n_samples=300)

# Train with custom architecture
model = ANN([2, 16, 8, 1], activation='relu', learning_rate=0.05)
model.train(X, y, epochs=1000, batch_size=32)
```

### Example 3: Visualization

```python
from visualizer import plot_decision_boundary, plot_training_curves

# Plot decision boundary
img = plot_decision_boundary(X, y, model, title="My ANN")

# Plot training curves
img = plot_training_curves(model.loss_history, model.accuracy_history)
```

## 🎛️ Configuration Options

### Network Architecture
- **Input Layer**: Automatically set to 2 (for 2D datasets)
- **Hidden Layers**: Configurable (e.g., `[8, 4]` for two hidden layers)
- **Output Layer**: 1 neuron for binary classification

### Hyperparameters
- **Learning Rate**: 0.001 - 0.5 (default: 0.1)
- **Epochs**: 100 - 2000 (default: 500)
- **Batch Size**: 16 - 128 (default: 32)
- **Activation**: sigmoid, tanh, relu (default: tanh)

## 📊 Understanding the Visualizations

### Decision Boundary Plot
- **Blue circles**: Class 0 samples
- **Red squares**: Class 1 samples
- **Background colors**: Predicted class probabilities
- **Black line**: Decision boundary (0.5 probability)

### Training Curves
- **Loss**: Should decrease over time
- **Accuracy**: Should increase over time

## 🧪 Testing

Test the ANN on the XOR problem:

```bash
python -c "from ann_core import ANN; from datasets import generate_xor; X, y = generate_xor(); ann = ANN([2, 4, 1]); ann.train(X, y, epochs=1000); print('Test Passed!')"
```

## 💡 Tips for Best Results

1. **Start Simple**: Begin with the XOR dataset to understand the basics
2. **Layer Sizes**: Use 2-3 hidden layers for complex datasets
3. **Learning Rate**: Lower rates (0.01-0.05) for stable training
4. **Epochs**: More epochs for better convergence (but watch for overfitting)
5. **Activation**: Try different activations - tanh often works well

## 🔬 Technical Details

### Implementation Highlights

- **Weight Initialization**: He initialization for better gradient flow
- **Mini-batch Training**: Efficient training with configurable batch size
- **Gradient Clipping**: Prevents numerical overflow in activations
- **Vectorized Operations**: NumPy for fast matrix computations

### Mathematical Foundation

**Forward Pass:**
```
z^[l] = W^[l] · a^[l-1] + b^[l]
a^[l] = activation(z^[l])
```

**Backward Pass:**
```
δ^[l] = (W^[l+1])^T · δ^[l+1] ⊙ activation'(z^[l])
dW^[l] = δ^[l] · (a^[l-1])^T
db^[l] = δ^[l]
```

## 📝 License

MIT License - feel free to use this for learning and experimentation!

## 🤝 Contributing

This is a demo project for educational purposes. Feel free to extend it with:
- Multi-class classification
- More activation functions
- Regularization techniques
- Different optimizers (Adam, RMSprop)
- Convolutional layers

## 📚 References

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)

## 🎉 Acknowledgments

Built with:
- **NumPy**: Numerical computations
- **Matplotlib**: Plotting and visualization
- **Gradio**: Interactive web interface
- **scikit-learn**: Dataset generation

---

**Happy Learning! 🚀**
