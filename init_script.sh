#!/bin/bash

echo "=========================================="
echo "Installing OpenCV System Dependencies"
echo "=========================================="

# Update package list
sudo apt-get update

# Install OpenCV dependencies
sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0

echo ""
echo "=========================================="
echo "✓ Dependencies installed!"
echo "=========================================="
echo ""
echo "Now try running again:"
echo "  streamlit run webcam_test.py"
echo ""
EOF
chmod +x /mnt/user-data/outputs/streamlit-transfer-learning/fix_opencv.sh
cat /mnt/user-data/outputs/streamlit-transfer-learning/fix_opencv.sh
Output

#!/bin/bash

echo "=========================================="
echo "Installing OpenCV System Dependencies"
echo "=========================================="

# Update package list
sudo apt-get update

# Install OpenCV dependencies
sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0

echo ""
echo "=========================================="
echo "✓ Dependencies installed!"
echo "=========================================="
echo ""
echo "Now try running again:"
echo "  streamlit run webcam_test.py"
echo ""

