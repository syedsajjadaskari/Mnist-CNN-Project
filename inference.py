import os
import sys
import config 
from models import MobileNetClassfier
from utils import extract_frames, preprocess_batch, print_results_summary

def classfify_video(video_path):
    if os.path.exists(video_path):
        print("The video not exits at path  ")
        return

    #extration of frames
    frames, _ = extract_frames(video_path, num_frames=NUM_FRAMES)

    #classfiy the object
    classfier = MobileNetClassfier()
    preprocessing = preprocess_batch(frames, target_size = config.INPUT_SIZE)
    predictions = classfier.predict(preprocessing)

    #results
    print("results summmary")
    print_results_summary(predictions, len(frames))


if __name___ == '__main__':
    if len(sys.argv) < 2:
        print("python inference ")
        sys.exit(1)

    classfify_video(sys.argv[1])
    


