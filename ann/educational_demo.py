"""
Educational ANN Demo - Understanding Neural Network Internals
Shows weights, biases, calculations, and network structure step-by-step
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
from PIL import Image

# Simple neural network for demonstration
class SimpleANN:
    def __init__(self, input_size=2, hidden_size=3, output_size=1):
        """Create a simple 3-layer network"""
        np.random.seed(42)
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        # Store for visualization
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        """Forward propagation with step tracking"""
        # Layer 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2


def create_network_visualization(input_vals, hidden_size, show_weights=True):
    """Visualize the network structure with weights and biases"""
    
    # Create network
    model = SimpleANN(input_size=2, hidden_size=hidden_size, output_size=1)
    
    # Forward pass
    X = np.array([[input_vals[0], input_vals[1]]])
    output = model.forward(X)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    
    # Main network diagram
    ax_main = plt.subplot(2, 2, (1, 3))
    ax_main.set_xlim(0, 10)
    ax_main.set_ylim(0, max(hidden_size + 2, 6))
    ax_main.axis('off')
    
    # Layer positions
    input_x, hidden_x, output_x = 1, 5, 9
    
    # Draw input layer
    input_y = [2, 3]
    for i, y in enumerate(input_y):
        circle = plt.Circle((input_x, y), 0.3, color='lightblue', ec='black', linewidth=2, zorder=3)
        ax_main.add_patch(circle)
        ax_main.text(input_x, y, f'{input_vals[i]:.2f}', ha='center', va='center', fontweight='bold', fontsize=10)
        ax_main.text(input_x - 0.7, y, f'x{i+1}', ha='right', va='center', fontsize=11, fontweight='bold')
    
    ax_main.text(input_x, 5, 'INPUT\nLAYER', ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Draw hidden layer
    hidden_y = np.linspace(1, hidden_size, hidden_size)
    for i, y in enumerate(hidden_y):
        color = plt.cm.RdYlGn(model.a1[0, i])
        circle = plt.Circle((hidden_x, y), 0.3, color=color, ec='black', linewidth=2, zorder=3)
        ax_main.add_patch(circle)
        ax_main.text(hidden_x, y, f'{model.a1[0, i]:.2f}', ha='center', va='center', 
                    fontweight='bold', fontsize=9)
    
    ax_main.text(hidden_x, hidden_size + 1.5, 'HIDDEN\nLAYER', ha='center', fontsize=12, 
                fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Draw output layer
    output_y = 2.5
    circle = plt.Circle((output_x, output_y), 0.3, color='lightcoral', ec='black', linewidth=2, zorder=3)
    ax_main.add_patch(circle)
    ax_main.text(output_x, output_y, f'{output[0,0]:.2f}', ha='center', va='center', 
                fontweight='bold', fontsize=10)
    ax_main.text(output_x + 0.7, output_y, 'Output', ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax_main.text(output_x, 5, 'OUTPUT\nLAYER', ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Draw connections with weights
    if show_weights:
        for i, iy in enumerate(input_y):
            for j, hy in enumerate(hidden_y):
                weight = model.W1[i, j]
                color = 'green' if weight > 0 else 'red'
                alpha = min(abs(weight), 1.0)
                ax_main.plot([input_x + 0.3, hidden_x - 0.3], [iy, hy], 
                           color=color, alpha=alpha, linewidth=2, zorder=1)
                # Show weight value
                mid_x, mid_y = (input_x + hidden_x) / 2, (iy + hy) / 2
                ax_main.text(mid_x, mid_y, f'{weight:.2f}', fontsize=7, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        for i, hy in enumerate(hidden_y):
            weight = model.W2[i, 0]
            color = 'green' if weight > 0 else 'red'
            alpha = min(abs(weight), 1.0)
            ax_main.plot([hidden_x + 0.3, output_x - 0.3], [hy, output_y], 
                       color=color, alpha=alpha, linewidth=2, zorder=1)
    
    ax_main.set_title('🧠 Neural Network Structure', fontsize=16, fontweight='bold', pad=20)
    
    # Weights matrix visualization
    ax_w1 = plt.subplot(2, 2, 2)
    im1 = ax_w1.imshow(model.W1.T, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax_w1.set_title('Weights: Input → Hidden (W1)', fontsize=11, fontweight='bold')
    ax_w1.set_xlabel('Input Neurons')
    ax_w1.set_ylabel('Hidden Neurons')
    plt.colorbar(im1, ax=ax_w1, label='Weight Value')
    
    # Add weight values as text
    for i in range(model.W1.shape[1]):
        for j in range(model.W1.shape[0]):
            ax_w1.text(j, i, f'{model.W1[j, i]:.2f}', ha='center', va='center', fontsize=9)
    
    # Weights matrix 2
    ax_w2 = plt.subplot(2, 2, 4)
    im2 = ax_w2.imshow(model.W2, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax_w2.set_title('Weights: Hidden → Output (W2)', fontsize=11, fontweight='bold')
    ax_w2.set_xlabel('Output Neurons')
    ax_w2.set_ylabel('Hidden Neurons')
    plt.colorbar(im2, ax=ax_w2, label='Weight Value')
    
    # Add weight values
    for i in range(model.W2.shape[0]):
        for j in range(model.W2.shape[1]):
            ax_w2.text(j, i, f'{model.W2[i, j]:.2f}', ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    # Create calculation explanation
    calc_text = f"""
## 🔢 Step-by-Step Calculations

### Input Values
- x₁ = {input_vals[0]:.2f}
- x₂ = {input_vals[1]:.2f}

---

### Layer 1: Input → Hidden

**Formula:** `z = (input × weights) + bias`  
**Then:** `activation = sigmoid(z)`

"""
    
    for i in range(hidden_size):
        w1_i1 = model.W1[0, i]
        w1_i2 = model.W1[1, i]
        b1_i = model.b1[0, i]
        z1_i = model.z1[0, i]
        a1_i = model.a1[0, i]
        
        calc_text += f"""
**Hidden Neuron {i+1}:**
- z₁[{i+1}] = (x₁ × w₁[{i+1}]) + (x₂ × w₂[{i+1}]) + bias
- z₁[{i+1}] = ({input_vals[0]:.2f} × {w1_i1:.2f}) + ({input_vals[1]:.2f} × {w1_i2:.2f}) + {b1_i:.2f}
- z₁[{i+1}] = {z1_i:.4f}
- **activation₁[{i+1}] = sigmoid({z1_i:.4f}) = {a1_i:.4f}** ✅

"""
    
    calc_text += """
---

### Layer 2: Hidden → Output

**Formula:** Same process with hidden layer activations as input

"""
    
    calc_text += f"""
**Output Neuron:**
"""
    for i in range(hidden_size):
        calc_text += f"- (hidden[{i+1}] × w[{i+1}]) = ({model.a1[0,i]:.4f} × {model.W2[i,0]:.2f}) = {model.a1[0,i] * model.W2[i,0]:.4f}\n"
    
    calc_text += f"""
- Sum + bias = {model.z2[0,0]:.4f}
- **Final Output = sigmoid({model.z2[0,0]:.4f}) = {output[0,0]:.4f}** 🎯

---

### 📚 Key Concepts

**Weights (W):**
- Numbers that determine how much each input matters
- Green connections = positive weights (increase signal)
- Red connections = negative weights (decrease signal)
- Larger absolute value = stronger influence

**Biases (b):**
- Added to shift the activation
- Help the network learn better patterns
- Current biases: {model.b1[0,0]:.2f} (hidden), {model.b2[0,0]:.2f} (output)

**Sigmoid Activation:**
- Squashes any number to range [0, 1]
- Formula: σ(x) = 1 / (1 + e⁻ˣ)
- Makes the network non-linear!

**Forward Propagation:**
- Data flows from left to right
- Each layer transforms the data
- Final output is the prediction
"""
    
    return img, calc_text


# Create Gradio interface
with gr.Blocks(title="🧠 ANN Internals - Educational Demo") as demo:
    
    gr.Markdown("""
    # 🧠 Understanding Neural Network Internals
    
    ## Learn How ANNs Really Work!
    
    This demo shows you **exactly** what happens inside a neural network:
    - ⚖️ **Weights**: How much each connection matters
    - ➕ **Biases**: The adjustment values
    - 🔢 **Calculations**: The actual math step-by-step
    - 🏗️ **Structure**: How layers connect
    
    ### Try it yourself! 👇
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 🎛️ Controls")
            
            input1 = gr.Slider(-2, 2, value=0.5, step=0.1, label="Input 1 (x₁)", 
                              info="First input value")
            input2 = gr.Slider(-2, 2, value=1.0, step=0.1, label="Input 2 (x₂)", 
                              info="Second input value")
            
            hidden_neurons = gr.Slider(2, 5, value=3, step=1, label="Hidden Layer Size",
                                      info="Number of neurons in hidden layer")
            
            show_weights_check = gr.Checkbox(value=True, label="Show Weight Values on Connections")
            
            compute_btn = gr.Button("🚀 Compute Forward Pass", variant="primary", size="lg")
            
            gr.Markdown("""
            ---
            ### 💡 What to Try:
            
            1. **Change inputs** - See how output changes
            2. **Add more neurons** - More learning capacity
            3. **Watch the colors** - Neuron activation levels
            4. **Read calculations** - Understand the math
            
            ### 🎨 Color Guide:
            - **Green lines** = Positive weights
            - **Red lines** = Negative weights
            - **Thicker lines** = Stronger weights
            - **Neuron colors** = Activation level
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("## 📊 Network Visualization")
            
            with gr.Tab("🏗️ Network Structure"):
                network_img = gr.Image(label="Neural Network with Weights & Biases")
                
            with gr.Tab("🔢 Detailed Calculations"):
                calculations = gr.Markdown()
    
    # Examples
    gr.Markdown("## 💡 Try These Examples")
    gr.Examples(
        examples=[
            [0.5, 1.0, 3, True],
            [1.5, -0.5, 4, True],
            [-1.0, 1.5, 5, False],
        ],
        inputs=[input1, input2, hidden_neurons, show_weights_check],
        label="Click to load example configurations"
    )
    
    gr.Markdown("""
    ---
    
    ## 📖 Understanding the Components
    
    ### 🏗️ Network Architecture
    - **Input Layer**: Receives your data (2 values in this demo)
    - **Hidden Layer**: Learns patterns (adjustable size)
    - **Output Layer**: Makes the final prediction (1 value)
    
    ### ⚖️ Weights (W)
    - Connect neurons between layers
    - **Positive weights**: Amplify the signal
    - **Negative weights**: Reduce the signal
    - **Magnitude**: How strong the connection is
    - Shown as colored lines and in matrices
    
    ### ➕ Biases (b)
    - One per neuron
    - Shifts the activation function
    - Helps network learn better
    - Added after weight multiplication
    
    ### 🔢 Forward Propagation
    1. **Multiply** inputs by weights
    2. **Add** all weighted inputs together
    3. **Add** the bias
    4. **Apply** activation function (sigmoid)
    5. **Repeat** for next layer
    
    ### 📈 Sigmoid Function
    - Converts any number to range [0, 1]
    - Formula: `1 / (1 + e^(-x))`
    - Creates non-linearity (enables learning complex patterns)
    
    ---
    
    **🎓 Perfect for learning the fundamentals of neural networks!**
    """)
    
    # Connect button
    compute_btn.click(
        fn=lambda i1, i2, h, w: create_network_visualization([i1, i2], int(h), w),
        inputs=[input1, input2, hidden_neurons, show_weights_check],
        outputs=[network_img, calculations]
    )
    
    # Also update on input change
    for inp in [input1, input2, hidden_neurons, show_weights_check]:
        inp.change(
            fn=lambda i1, i2, h, w: create_network_visualization([i1, i2], int(h), w),
            inputs=[input1, input2, hidden_neurons, show_weights_check],
            outputs=[network_img, calculations]
        )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧠 ANN Internals - Educational Demo")
    print("="*60)
    print("\n📚 Learn about weights, biases, and calculations!")
    print("🌐 Opening in your browser...\n")
    
    demo.launch(share=False, server_name="127.0.0.1", server_port=7863)
