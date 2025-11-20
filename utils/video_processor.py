"""
Video processing utilities
"""
import cv2
import numpy as np
from tqdm import tqdm


def extract_frames(video_path, num_frames=30, frame_skip=1):
    """
    Extract frames from video at regular intervals
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        frame_skip: Skip every nth frame for faster processing
    
    Returns:
        frames: List of numpy arrays (RGB format)
        video_info: Dictionary with video metadata
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    video_info = {
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'duration': duration
    }
    
    print(f"\nVideo Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {duration:.2f} seconds")
    
    # Calculate frame indices to extract
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in tqdm(frame_indices, desc="Extracting frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    
    return frames, video_info


def get_video_info(video_path):
    """
    Get video metadata without extracting frames
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video metadata
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    info = {
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    
    info['duration'] = info['total_frames'] / info['fps'] if info['fps'] > 0 else 0
    
    cap.release()
    return info


def save_frame(frame, output_path):
    """
    Save a single frame to disk
    
    Args:
        frame: Numpy array (RGB format)
        output_path: Path to save the frame
    """
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, frame_bgr)