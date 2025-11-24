"""
Simple ANN Demo for Beginners
An educational, interactive demonstration of how neural networks learn
Perfect for students with minimal background in AI/ML
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from ann_core import ANN
from datasets import generate_xor, generate_circles, generate_moons
import io
from PIL import Image


def create_simple_plot(X, y, title="Dataset"):
    """Create a simple scatter plot of the data."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    y_flat = y.flatten()
    ax.scatter(X[y_flat == 0, 0], X[y_flat == 0, 1], 
              c='blue', marker='o', s=150, edgecolors='black', 
              linewidths=2, label='🔵 Class 0 (Blue)', alpha=0.8)
    ax.scatter(X[y_flat == 1, 0], X[y_flat == 1, 1], 
              c='red', marker='s', s=150, edgecolors='black', 
              linewidths=2, label='🔴 Class 1 (Red)', alpha=0.8)
    
    ax.set_xlabel('Feature 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img


def train_simple_ann(dataset_choice, num_neurons, epochs_choice):
    """
    Train a simple ANN and show results with explanations.
    """
    # Generate dataset
    if dataset_choice == "🎯 XOR Pattern (Hardest)":
        X, y = generate_xor(n_samples=200)
        dataset_name = "XOR"
        explanation = """
### 🎯 XOR Pattern - The Classic Challenge!

**What is XOR?**
XOR (exclusive OR) means "one or the other, but not both":
- Blue points (0): Top-left AND Bottom-right
- Red points (1): Top-right AND Bottom-left

**Why is it hard?**
You can't draw a single straight line to separate blue from red!
This is why we need a neural network with hidden layers.
"""
    elif dataset_choice == "⭕ Circles (Medium)":
        X, y = generate_circles(n_samples=200)
        dataset_name = "Circles"
        explanation = """
### ⭕ Circles - Learning Curved Boundaries

**What do you see?**
- Blue points (0): Outer circle
- Red points (1): Inner circle

**The Challenge:**
The boundary between classes is circular, not straight!
The neural network needs to learn this curved pattern.
"""
    else:  # Moons
        X, y = generate_moons(n_samples=200)
        dataset_name = "Moons"
        explanation = """
### 🌙 Moons - Two Crescents

**What do you see?**
- Blue points (0): Upper crescent
- Red points (1): Lower crescent

**The Challenge:**
The classes are interleaved like two moons.
A straight line can't separate them!
"""
    
    # Show original data
    data_plot = create_simple_plot(X, y, f"{dataset_name} Dataset - Before Training")
    
    # Map epochs choice to actual number
    epochs_map = {
        "Quick (100 epochs)": 100,
        "Normal (500 epochs)": 500,
        "Thorough (1000 epochs)": 1000
    }
    epochs = epochs_map[epochs_choice]
    
    # Map neurons choice
    neurons_map = {
        "Small (4 neurons)": 4,
        "Medium (8 neurons)": 8,
        "Large (16 neurons)": 16
    }
    neurons = neurons_map[num_neurons]
    
    # Create and train model
    model = ANN([2, neurons, 1], activation='tanh', learning_rate=0.1)
    
    status_text = f"""
## 🚀 Training Started!

**Network Architecture:**
- Input Layer: 2 neurons (for x and y coordinates)
- Hidden Layer: {neurons} neurons (learning patterns)
- Output Layer: 1 neuron (predicting class)

**Training for {epochs} epochs...**
"""
    
    # Train the model
    loss_history, accuracy_history = model.train(X, y, epochs=epochs, batch_size=32, verbose=False)
    
    # Create result visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Decision Boundary
    xx, yy, Z = model.get_decision_boundary(X, resolution=200)
    contour = ax1.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3)
    
    y_flat = y.flatten()
    ax1.scatter(X[y_flat == 0, 0], X[y_flat == 0, 1], 
               c='blue', marker='o', s=100, edgecolors='black', 
               linewidths=1.5, label='Class 0', alpha=0.9)
    ax1.scatter(X[y_flat == 1, 0], X[y_flat == 1, 1], 
               c='red', marker='s', s=100, edgecolors='black', 
               linewidths=1.5, label='Class 1', alpha=0.9)
    
    ax1.set_title('✅ What the Network Learned\n(Decision Boundary)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Feature 1', fontsize=11)
    ax1.set_ylabel('Feature 2', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss Over Time
    ax2.plot(loss_history, 'b-', linewidth=3)
    ax2.fill_between(range(len(loss_history)), loss_history, alpha=0.3)
    ax2.set_title('📉 Loss Over Time\n(Lower is Better!)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch (Training Round)', fontsize=11)
    ax2.set_ylabel('Loss (Error)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # 3. Accuracy Over Time
    ax3.plot(accuracy_history, 'g-', linewidth=3)
    ax3.fill_between(range(len(accuracy_history)), accuracy_history, alpha=0.3)
    ax3.set_title('📈 Accuracy Over Time\n(Higher is Better!)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Epoch (Training Round)', fontsize=11)
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # Add percentage labels
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    
    # 4. Network Diagram
    ax4.axis('off')
    ax4.text(0.5, 0.9, '🧠 Your Neural Network', 
            ha='center', fontsize=14, fontweight='bold', transform=ax4.transAxes)
    
    layers_text = f"""
    Input Layer          Hidden Layer         Output Layer
    (2 neurons)          ({neurons} neurons)          (1 neuron)
    
         ●                    ●                    ●
         ●                    ●                 
                              ●                 Predicts:
      [x, y]                 ...               0 or 1
    coordinates           Learning            (Blue/Red)
                          patterns!
    
    ➡️  Forward Pass: Data flows left to right
    ⬅️  Backward Pass: Learning flows right to left
    """
    
    ax4.text(0.5, 0.5, layers_text, ha='center', va='center', 
            fontsize=11, family='monospace', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    result_img = Image.open(buf)
    plt.close(fig)
    
    # Final metrics
    final_loss = loss_history[-1]
    final_accuracy = accuracy_history[-1]
    
    results_text = f"""
{explanation}

---

## 🎓 Training Results

### Final Performance
- **Accuracy: {final_accuracy*100:.1f}%** {'🎉 Excellent!' if final_accuracy > 0.9 else '👍 Good!' if final_accuracy > 0.8 else '📚 Keep learning!'}
- **Loss: {final_loss:.4f}** (Lower is better)

### What Happened?
1. **Started**: The network had random weights (didn't know anything)
2. **Learning**: Over {epochs} epochs, it adjusted its weights
3. **Result**: Now it can separate the classes!

### Understanding the Plots:

**📊 Top-Left (Decision Boundary):**
- Background color shows what the network predicts
- Black line is the decision boundary
- The network learned to separate blue from red!

**📉 Top-Right (Loss):**
- Shows how wrong the network was over time
- Going down = getting better!
- Flat line at the end = finished learning

**📈 Bottom-Left (Accuracy):**
- Shows how many points it got right
- Going up = getting better!
- {final_accuracy*100:.1f}% means it correctly classified {final_accuracy*100:.1f}% of points

**🧠 Bottom-Right (Network Structure):**
- Shows how your network is organized
- More neurons = more learning capacity
- But too many can be overkill!

---

### 💡 Try This Next:
1. Change the number of neurons - see what happens!
2. Try different datasets - which is hardest?
3. Use more epochs - does accuracy improve?
"""
    
    return data_plot, result_img, results_text


# Create the Gradio interface
with gr.Blocks(title="🧠 ANN for Beginners") as demo:
    
    gr.Markdown("""
    # 🧠 Artificial Neural Network (ANN) - Learn by Doing!
    
    ### Welcome! 👋
    
    This is an **interactive learning tool** to help you understand how neural networks work.
    
    **No prior knowledge needed!** Just follow these steps:
    
    1. **Choose a dataset** - Pick a pattern for the network to learn
    2. **Set the network size** - How many "neurons" should it use?
    3. **Choose training time** - How long should it practice?
    4. **Click "Train"** - Watch the magic happen! ✨
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Settings")
            
            dataset = gr.Radio(
                choices=[
                    "🌙 Moons (Easy)",
                    "⭕ Circles (Medium)",
                    "🎯 XOR Pattern (Hardest)"
                ],
                value="🌙 Moons (Easy)",
                label="1️⃣ Choose Your Dataset",
                info="Start with Moons if you're new!"
            )
            
            neurons = gr.Radio(
                choices=[
                    "Small (4 neurons)",
                    "Medium (8 neurons)",
                    "Large (16 neurons)"
                ],
                value="Medium (8 neurons)",
                label="2️⃣ Network Size",
                info="More neurons = more learning power"
            )
            
            epochs = gr.Radio(
                choices=[
                    "Quick (100 epochs)",
                    "Normal (500 epochs)",
                    "Thorough (1000 epochs)"
                ],
                value="Normal (500 epochs)",
                label="3️⃣ Training Time",
                info="More epochs = more practice"
            )
            
            train_btn = gr.Button("🚀 Train the Network!", variant="primary", size="lg")
            
            gr.Markdown("""
            ---
            ### 📚 Quick Guide
            
            **What is a Neural Network?**
            Think of it as a pattern-learning machine!
            
            **How does it learn?**
            1. Makes a guess
            2. Checks if it's right
            3. Adjusts itself
            4. Repeats thousands of times!
            
            **What are epochs?**
            One epoch = one complete pass through all the data
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("## 📊 Results")
            
            with gr.Tab("📍 Original Data"):
                data_img = gr.Image(label="The Dataset")
                
            with gr.Tab("🎯 Training Results"):
                result_img = gr.Image(label="What the Network Learned")
                
            with gr.Tab("📖 Explanation"):
                explanation = gr.Markdown()
    
    # Examples section
    gr.Markdown("""
    ---
    ## 💡 Suggested Experiments
    
    Try these step-by-step to learn:
    """)
    
    gr.Examples(
        examples=[
            ["🌙 Moons (Easy)", "Small (4 neurons)", "Quick (100 epochs)"],
            ["🌙 Moons (Easy)", "Medium (8 neurons)", "Normal (500 epochs)"],
            ["⭕ Circles (Medium)", "Medium (8 neurons)", "Normal (500 epochs)"],
            ["🎯 XOR Pattern (Hardest)", "Large (16 neurons)", "Thorough (1000 epochs)"],
        ],
        inputs=[dataset, neurons, epochs],
        label="Click any row to try it!"
    )
    
    gr.Markdown("""
    ---
    
    ### 🎓 Learning Objectives
    
    After using this demo, you should understand:
    
    ✅ Neural networks learn patterns from data  
    ✅ They improve through practice (epochs)  
    ✅ More neurons can learn more complex patterns  
    ✅ Training shows progress through loss and accuracy  
    ✅ Decision boundaries show what the network learned  
    
    ### 🤔 Common Questions
    
    **Q: Why doesn't it reach 100% accuracy?**  
    A: Some patterns are hard! Also, we add noise to make it realistic.
    
    **Q: What if accuracy goes down?**  
    A: Try more epochs or more neurons!
    
    **Q: Which dataset is hardest?**  
    A: XOR! It's the classic challenge for neural networks.
    
    **Q: What's the black line in the results?**  
    A: That's the decision boundary - where the network decides "blue vs red"
    
    ---
    
    **Made with ❤️ for students learning AI**
    """)
    
    # Connect the button
    train_btn.click(
        fn=train_simple_ann,
        inputs=[dataset, neurons, epochs],
        outputs=[data_img, result_img, explanation]
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧠 Simple ANN Demo for Beginners")
    print("="*60)
    print("\n📚 Perfect for students learning about neural networks!")
    print("🌐 Opening in your browser...\n")
    
    demo.launch(share=False, server_name="127.0.0.1", server_port=7861)
