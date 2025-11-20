#!/bin/bash

# Create virtual environment
python3 -m venv env

# Activate the environment
source env/bin/activate

# Install packages (add your own here)
pip install -r requirements_dev.txt
# Freeze requirements
pip freeze > project_library.txt

echo "Environment created and activated."
echo "requirements.txt generated."
