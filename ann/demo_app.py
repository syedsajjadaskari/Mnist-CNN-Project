"""
Interactive ANN Demo using Gradio
A web-based interface for training and visualizing neural networks
"""

import gradio as gr
import numpy as np
from ann_core import ANN
from datasets import get_dataset, DATASETS
from visualizer import (
    plot_decision_boundary, 
    plot_training_curves, 
    plot_network_architecture,
    create_comparison_plot
)


def train_and_visualize(
    dataset_name: str,
    hidden_layers: str,
    learning_rate: float,
    epochs: int,
    activation: str,
    n_samples: int
):
    """
    Train an ANN and generate visualizations.
    
    Args:
        dataset_name: Name of the dataset to use
        hidden_layers: Comma-separated list of hidden layer sizes
        learning_rate: Learning rate for training
        epochs: Number of training epochs
        activation: Activation function name
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (comparison_plot, architecture_plot, metrics_text)
    """
    try:
        # Parse hidden layers
        hidden_sizes = [int(x.strip()) for x in hidden_layers.split(',') if x.strip()]
        
        # Generate dataset
        X, y = get_dataset(dataset_name, n_samples=n_samples)
        
        # Create model architecture
        layer_sizes = [2] + hidden_sizes + [1]
        
        # Initialize and train model
        model = ANN(layer_sizes, activation=activation, learning_rate=learning_rate)
        
        print(f"\n{'='*60}")
        print(f"Training ANN on {dataset_name} dataset")
        print(f"Architecture: {layer_sizes}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Activation: {activation}")
        print(f"{'='*60}\n")
        
        loss_history, accuracy_history = model.train(X, y, epochs=epochs, batch_size=32, verbose=True)
        
        # Generate visualizations
        comparison_plot = create_comparison_plot(X, y, model, loss_history, accuracy_history)
        architecture_plot = plot_network_architecture(layer_sizes)
        
        # Generate metrics text
        final_loss = loss_history[-1]
        final_accuracy = accuracy_history[-1]
        
        metrics_text = f"""
## Training Results

**Dataset:** {dataset_name}  
**Architecture:** {' → '.join(map(str, layer_sizes))}  
**Activation:** {activation}  
**Learning Rate:** {learning_rate}  
**Epochs:** {epochs}  

### Final Metrics
- **Loss:** {final_loss:.4f}
- **Accuracy:** {final_accuracy:.4f} ({final_accuracy*100:.2f}%)

### Training Summary
- **Initial Loss:** {loss_history[0]:.4f}
- **Final Loss:** {final_loss:.4f}
- **Loss Reduction:** {((loss_history[0] - final_loss) / loss_history[0] * 100):.2f}%
- **Initial Accuracy:** {accuracy_history[0]:.4f}
- **Final Accuracy:** {final_accuracy:.4f}
- **Accuracy Improvement:** {((final_accuracy - accuracy_history[0]) * 100):.2f}%

✅ **Training Completed Successfully!**
"""
        
        return comparison_plot, architecture_plot, metrics_text
        
    except Exception as e:
        error_msg = f"❌ **Error:** {str(e)}\n\nPlease check your parameters and try again."
        return None, None, error_msg


# Create Gradio interface
with gr.Blocks(title="ANN Demo - Interactive Neural Network Trainer", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🧠 Artificial Neural Network (ANN) Demo
    
    Train a neural network from scratch and visualize its decision boundaries in real-time!
    
    This interactive demo allows you to:
    - Choose from various datasets (XOR, Circles, Moons, Spiral, etc.)
    - Configure network architecture and hyperparameters
    - Visualize decision boundaries and training progress
    - Experiment with different activation functions
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Dataset & Model Configuration")
            
            dataset_dropdown = gr.Dropdown(
                choices=list(DATASETS.keys()),
                value='XOR',
                label="Dataset",
                info="Select the dataset to train on"
            )
            
            n_samples_slider = gr.Slider(
                minimum=100,
                maximum=500,
                value=200,
                step=50,
                label="Number of Samples",
                info="More samples = better visualization but slower training"
            )
            
            hidden_layers_text = gr.Textbox(
                value="8, 4",
                label="Hidden Layers",
                info="Comma-separated layer sizes (e.g., '8, 4' for two hidden layers)"
            )
            
            activation_dropdown = gr.Dropdown(
                choices=['sigmoid', 'tanh', 'relu'],
                value='tanh',
                label="Activation Function",
                info="Activation function for hidden layers"
            )
            
            learning_rate_slider = gr.Slider(
                minimum=0.001,
                maximum=0.5,
                value=0.1,
                step=0.001,
                label="Learning Rate",
                info="Higher = faster learning but less stable"
            )
            
            epochs_slider = gr.Slider(
                minimum=100,
                maximum=2000,
                value=500,
                step=100,
                label="Training Epochs",
                info="Number of training iterations"
            )
            
            train_button = gr.Button("🚀 Train Network", variant="primary", size="lg")
            
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results & Visualizations")
            
            with gr.Tab("Training Results"):
                comparison_plot = gr.Image(label="Decision Boundary & Training Curves")
                
            with gr.Tab("Network Architecture"):
                architecture_plot = gr.Image(label="Network Structure")
                
            with gr.Tab("Metrics"):
                metrics_output = gr.Markdown()
    
    # Examples section
    gr.Markdown("### 💡 Try These Examples")
    
    gr.Examples(
        examples=[
            ["XOR", "8, 4", 0.1, 500, "tanh", 200],
            ["Circles", "16, 8", 0.05, 1000, "relu", 300],
            ["Moons", "12, 6", 0.1, 800, "tanh", 200],
            ["Spiral", "32, 16, 8", 0.01, 1500, "relu", 300],
            ["Linear", "4", 0.1, 300, "sigmoid", 200],
        ],
        inputs=[dataset_dropdown, hidden_layers_text, learning_rate_slider, 
                epochs_slider, activation_dropdown, n_samples_slider],
        label="Click to load example configurations"
    )
    
    # Connect the train button
    train_button.click(
        fn=train_and_visualize,
        inputs=[dataset_dropdown, hidden_layers_text, learning_rate_slider, 
                epochs_slider, activation_dropdown, n_samples_slider],
        outputs=[comparison_plot, architecture_plot, metrics_output]
    )
    
    gr.Markdown("""
    ---
    ### 📚 About This Demo
    
    This demo implements a **feedforward neural network from scratch** using only NumPy. 
    It demonstrates:
    
    - **Forward Propagation:** Computing predictions through the network
    - **Backpropagation:** Computing gradients and updating weights
    - **Decision Boundaries:** Visualizing what the network has learned
    - **Training Dynamics:** Observing loss and accuracy over time
    
    **Tips for Best Results:**
    - Start with the XOR dataset to see non-linear classification
    - Use 2-3 hidden layers for complex datasets like Spiral
    - Lower learning rates for more stable training
    - More epochs for better convergence
    
    Built with ❤️ using NumPy, Matplotlib, and Gradio
    """)


if __name__ == "__main__":
    print("🚀 Launching ANN Demo...")
    print("📊 Open the URL below in your browser to interact with the demo")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
