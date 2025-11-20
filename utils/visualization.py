"""
Visualization utilities for results and analysis
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os


def plot_sample_frames(frames, predictions=None, num_display=6, save_path=None):
    num_display = min(num_display, len(frames))
    cols = 3
    rows = (num_display + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.ravel() if num_display > 1 else [axes]
    
    indices = np.linspace(0, len(frames) - 1, num_display, dtype=int)
    
    for i, idx in enumerate(indices):
        if i < len(axes):
            axes[i].imshow(frames[idx])
            axes[i].axis('off')
            
            if predictions and idx < len(predictions):
                top_pred = predictions[idx][0]
                title = f"{top_pred[0]}\n{top_pred[1]:.1%}"
                axes[i].set_title(title, fontsize=10, fontweight='bold')
    
    # Hide unused subplots
    for i in range(num_display, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_prediction_distribution(all_predictions, top_n=10, save_path=None):

    # Collect top predictions from each frame
    top_classes = [pred[0][0] for pred in all_predictions]
    class_counts = Counter(top_classes)
    
    # Get top N most common
    most_common = class_counts.most_common(top_n)
    classes = [item[0] for item in most_common]
    counts = [item[1] for item in most_common]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(classes, counts, color='skyblue', edgecolor='navy')
    
    # Add count labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f' {int(width)}', ha='left', va='center', fontweight='bold')
    
    ax.set_xlabel('Number of Frames', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Predicted Classes Across All Frames', 
                fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_confidence_timeline(all_predictions, save_path=None):
    confidences = [pred[0][1] for pred in all_predictions]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(confidences, marker='o', markersize=4, linewidth=2, 
            color='navy', markerfacecolor='skyblue')
    ax.fill_between(range(len(confidences)), confidences, alpha=0.3)
    
    ax.set_xlabel('Frame Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
    ax.set_title('Top Prediction Confidence Over Time', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def print_results_summary(all_predictions, num_frames, top_n=5):

    print("\n" + "="*70)
    print("VIDEO CLASSIFICATION RESULTS".center(70))
    print("="*70)
    
    # Most frequent predictions
    top_predictions = [pred[0][0] for pred in all_predictions]
    prediction_counts = Counter(top_predictions)
    most_common = prediction_counts.most_common(top_n)
    
    print("\n📊 Most Frequent Classifications:")
    for i, (class_name, count) in enumerate(most_common, 1):
        percentage = (count / num_frames) * 100
        print(f"  {i}. {class_name:<40} {count:>3}/{num_frames} frames ({percentage:>5.1f}%)")
    
    # Average confidence scores
    class_scores = {}
    for predictions in all_predictions:
        for class_name, prob in predictions:
            if class_name not in class_scores:
                class_scores[class_name] = []
            class_scores[class_name].append(prob)
    
    avg_class_scores = {k: np.mean(v) for k, v in class_scores.items()}
    top_by_confidence = sorted(avg_class_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    print("\n🎯 Top Classes by Average Confidence:")
    for i, (class_name, avg_conf) in enumerate(top_by_confidence, 1):
        print(f"  {i}. {class_name:<40} {avg_conf:>6.1%} avg confidence")
    
    # Final verdict
    final_prediction = most_common[0][0]
    final_confidence = avg_class_scores[final_prediction]
    
    print("\n" + "="*70)
    print(f"🏆 FINAL CLASSIFICATION: {final_prediction.upper()}".center(70))
    print(f"Confidence: {final_confidence:.1%}".center(70))
    print("="*70 + "\n")
    
    return {
        'final_prediction': final_prediction,
        'final_confidence': final_confidence,
        'top_predictions': most_common,
        'avg_confidences': top_by_confidence
    }