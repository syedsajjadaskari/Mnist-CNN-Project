# app.py
"""
Simple Video Object Detection App using Gradio
Upload a video and get real-time object detection with bounding boxes
"""

import gradio as gr
import cv2
import numpy as np
from tensorflow import keras
import tempfile
import os
from PIL import Image
import matplotlib.pyplot as plt

# Class names for CIFAR-10
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Color palette for bounding boxes (BGR format for OpenCV)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0)
]

# Global model variable
model = None

def load_model():
    """Load the trained model"""
    global model
    try:
        model = keras.models.load_model('models/best_model_fine_tuned.h5')
        return "✅ Model loaded successfully!"
    except Exception as e:
        try:
            model = keras.models.load_model('models/best_model.h5')
            return "✅ Model loaded successfully (base model)!"
        except:
            return "⚠️ Error: Model not found! Please train the model first using: python main.py"

def preprocess_frame(frame):
    """Preprocess frame for model prediction"""
    resized = cv2.resize(frame, (224, 224))
    normalized = resized / 255.0
    batched = np.expand_dims(normalized, axis=0)
    return batched

def predict_frame(frame):
    """Predict objects in a frame"""
    if model is None:
        return None, 0, 0
    
    processed = preprocess_frame(frame)
    predictions = model.predict(processed, verbose=0)
    
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    class_name = CLASS_NAMES[class_idx]
    
    return class_name, confidence, class_idx

def draw_prediction(frame, class_name, confidence, color):
    """Draw prediction on frame"""
    height, width = frame.shape[:2]
    
    # Draw semi-transparent overlay at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Draw text
    text = f"{class_name}: {confidence:.2%}"
    cv2.putText(frame, text, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    # Draw border
    cv2.rectangle(frame, (5, 5), (width-5, height-5), color, 5)
    
    return frame

def create_summary_plot(detections):
    """Create a bar chart of detections"""
    if not detections:
        return None
    
    # Sort by count
    sorted_items = sorted(detections.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, counts, color='steelblue', edgecolor='black', linewidth=1.5)
    
    # Customize
    ax.set_xlabel('Object Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frame Count', fontsize=12, fontweight='bold')
    ax.set_title('Object Detection Summary', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def process_video(video_path, skip_frames, progress=gr.Progress()):
    """Process video and detect objects"""
    
    if model is None:
        yield None, None, "⚠️ Please load the model first!", None
        return
    
    if video_path is None:
        yield None, None, "⚠️ Please upload a video first!", None
        return
    
    # Handle different video path formats
    if isinstance(video_path, dict):
        video_path = video_path.get('video', video_path.get('name', None))
    
    if not os.path.exists(str(video_path)):
        yield None, None, f"⚠️ Video file not found: {video_path}", None
        return
    
    print(f"\n{'='*60}")
    print(f"🎬 Starting Video Processing")
    print(f"{'='*60}")
    print(f"📁 Input: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video Info:")
    print(f"   - Resolution: {width}x{height}")
    print(f"   - FPS: {fps}")
    print(f"   - Total Frames: {total_frames}")
    print(f"   - Skip Frames: {skip_frames}")
    print(f"   - Frames to Analyze: {total_frames // skip_frames}")
    print(f"{'='*60}\n")
    
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detections = {}
    last_log_time = 0
    last_yield_frame = 0
    
    progress(0, desc="🚀 Starting video processing...")
    
    import time
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = frame.copy()
        
        # Process every Nth frame
        if frame_count % skip_frames == 0:
            class_name, confidence, class_idx = predict_frame(frame)
            
            if class_name:
                # Track detections
                if class_name in detections:
                    detections[class_name] += 1
                else:
                    detections[class_name] = 1
                
                # Draw on frame
                color = COLORS[class_idx]
                processed_frame = draw_prediction(processed_frame, class_name, confidence, color)
                
                # Log every 2 seconds
                current_time = time.time()
                if current_time - last_log_time >= 2:
                    elapsed = current_time - start_time
                    percent = (frame_count / total_frames) * 100
                    print(f"⏱️  {elapsed:.1f}s | Frame {frame_count}/{total_frames} ({percent:.1f}%) | Detected: {class_name} ({confidence:.1%})")
                    last_log_time = current_time
                
                # Yield preview every 30 frames
                if frame_count - last_yield_frame >= 30:
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Create interim summary
                    interim_summary = f"⏳ **Processing...**\n\n"
                    interim_summary += f"📊 Progress: {frame_count}/{total_frames} frames\n\n"
                    interim_summary += f"🎯 **Current Detections:**\n"
                    for cls, cnt in sorted(detections.items(), key=lambda x: x[1], reverse=True):
                        interim_summary += f"- {cls.capitalize()}: {cnt} frames\n"
                    
                    yield frame_rgb, None, interim_summary, None
                    last_yield_frame = frame_count
        
        # Write frame
        out.write(processed_frame)
        
        # Update progress
        frame_count += 1
        percent_done = frame_count / total_frames
        progress(percent_done, 
                desc=f"🎯 Frame {frame_count}/{total_frames} ({percent_done*100:.1f}%)")
    
    cap.release()
    out.release()
    
    elapsed_total = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Processing Complete!")
    print(f"⏱️  Total Time: {elapsed_total:.1f}s")
    print(f"📁 Output: {output_path}")
    print(f"{'='*60}\n")
    
    # Create summary text
    summary = f"📹 **Video Processed Successfully!**\n\n"
    summary += f"⏱️ **Processing Time:** {elapsed_total:.1f} seconds\n\n"
    summary += f"📊 **Video Info:**\n"
    summary += f"- Resolution: {width}x{height}\n"
    summary += f"- FPS: {fps}\n"
    summary += f"- Total Frames: {total_frames}\n"
    summary += f"- Frames Analyzed: {total_frames // skip_frames}\n\n"
    summary += f"🎯 **Final Detections:**\n"
    
    # Sort detections
    sorted_detections = sorted(detections.items(), key=lambda x: x[1], reverse=True)
    for class_name, count in sorted_detections:
        summary += f"- **{class_name.capitalize()}**: {count} frames\n"
    
    # Create plot
    plot = create_summary_plot(detections)
    
    # Return final frame preview
    cap2 = cv2.VideoCapture(output_path)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, final_frame = cap2.read()
    cap2.release()
    
    if ret:
        final_frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
    else:
        final_frame_rgb = None
    
    yield final_frame_rgb, output_path, summary, plot

# Create Gradio interface
def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Video Object Detection", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown(
            """
            # 🎥 Video Object Detection
            Upload a video and detect objects using CNN Transfer Learning
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Model loading
                with gr.Group():
                    gr.Markdown("### 🤖 Model Status")
                    load_btn = gr.Button("🔄 Load Model", variant="primary")
                    model_status = gr.Textbox(
                        label="Status",
                        value="Model not loaded. Click 'Load Model' button.",
                        interactive=False
                    )
                    load_btn.click(load_model, inputs=[], outputs=model_status)
                
                # Video upload
                with gr.Group():
                    gr.Markdown("### 📤 Upload Video")
                    video_input = gr.File(
                        label="Input Video",
                        file_types=["video"],
                        type="filepath"
                    )
                
                # Settings
                with gr.Group():
                    gr.Markdown("### ⚙️ Settings")
                    skip_frames = gr.Slider(
                        minimum=1,
                        maximum=30,
                        value=5,
                        step=1,
                        label="Process every Nth frame (Higher = Faster)",
                        info="Skip frames for faster processing"
                    )
                
                # Process button
                process_btn = gr.Button("🚀 Start Detection", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📚 Detectable Objects")
                gr.Markdown("""
                - ✈️ Airplane
                - 🚗 Automobile
                - 🐦 Bird
                - 🐱 Cat
                - 🦌 Deer
                - 🐕 Dog
                - 🐸 Frog
                - 🐴 Horse
                - 🚢 Ship
                - 🚚 Truck
                """)
                
                gr.Markdown("### 💡 Tips")
                gr.Markdown("""
                1. Train model first: `python main.py`
                2. Load model before processing
                3. Use shorter videos for faster results
                4. Adjust skip frames for speed vs accuracy
                """)
        
        gr.Markdown("---")
        
        # Output section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎬 Live Preview")
                live_preview = gr.Image(label="Current Frame", type="numpy")
                gr.Markdown("### 📹 Processed Video")
                video_output = gr.File(label="Download Processed Video", file_count="single")
                gr.Markdown("*Right-click and 'Save Link As...' to download*")
            
            with gr.Column():
                gr.Markdown("### 📊 Detection Summary")
                summary_output = gr.Markdown()
                plot_output = gr.Plot(label="Detection Chart")
        
        # Examples
        gr.Markdown("---")
        gr.Markdown("### 🎬 How It Works")
        gr.Markdown("""
        1. **Load Model**: Click the 'Load Model' button to load the trained CNN model
        2. **Upload Video**: Choose a video file (MP4, AVI, MOV, MKV)
        3. **Adjust Settings**: Set frame skip value (5 is recommended)
        4. **Start Detection**: Click 'Start Detection' to process the video
        5. **View Results**: See the processed video with bounding boxes and statistics
        6. **Download**: Download the processed video from the output player
        
        ⚠️ **Note**: Make sure you've trained the model first by running `python main.py`
        """)
        
        # Connect the process button
        process_btn.click(
            fn=process_video,
            inputs=[video_input, skip_frames],
            outputs=[video_output, summary_output, plot_output]
        )
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )