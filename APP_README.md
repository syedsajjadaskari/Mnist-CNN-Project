# Video Frame Classifier App

A Streamlit web application for classifying video frames using a trained CNN model.

## Features

- 🎥 **Video Upload**: Support for MP4, AVI, MOV, and MKV formats
- 🖼️ **Frame Extraction**: Configurable frame sampling rate
- 🤖 **Classification**: Uses trained MobileNetV2 transfer learning model
- 📊 **Visualization**: 
  - Sample frame predictions with confidence scores
  - Classification distribution bar chart
  - Statistics (total frames, average confidence, most common class)
- ⚙️ **Customizable**: Adjustable frame sampling rate and dataset selection

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train a model first (if not already done):
```bash
python main.py
```

This will create a trained model at `models/best_model.h5`.

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser (usually at `http://localhost:8501`)

3. Use the app:
   - Select your dataset/model from the sidebar
   - Adjust the frame sampling rate (default: 30 frames)
   - Upload a video file
   - Click "Classify Video"
   - View the results!

## How It Works

1. **Frame Extraction**: The app extracts frames from the uploaded video at the specified sampling rate (e.g., 1 frame every 30 frames)

2. **Preprocessing**: Each frame is:
   - Resized to the model's input size (224x224)
   - Normalized to [0, 1] range
   - Converted to RGB format

3. **Classification**: The trained CNN model predicts the class for each frame

4. **Visualization**: Results are displayed with:
   - Sample frames showing predictions and confidence scores
   - Distribution chart of all classifications
   - Summary statistics

## Supported Datasets

- **CIFAR-10**: 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **CIFAR-100**: 100 classes
- **Fashion MNIST**: 10 classes (clothing items)

## Configuration

You can modify settings in `config.py`:
- `IMG_SIZE`: Input image size for the model
- `MODEL_SAVE_PATH`: Path to the trained model
- `DATASET`: Default dataset selection

## Notes

- The app classifies individual frames, not objects within frames
- For best results, use videos containing content similar to your training dataset
- Higher frame sampling rates process more frames but take longer
- The model must be trained before using the app

## Troubleshooting

**Model not found error**: 
- Make sure you've trained a model by running `python main.py`
- Check that `models/best_model.h5` exists

**SSL Certificate Error**:
- The app includes SSL certificate fixes in the code
- If issues persist, try: `pip install --upgrade certifi`

**Video upload fails**:
- Ensure your video format is supported (MP4, AVI, MOV, MKV)
- Try converting your video to MP4 format

## Example

Upload a video of animals, vehicles, or objects matching your training dataset to see the classifier in action!
