#!/bin/bash

# Setup script for reward function learning system

echo "Setting up Reward Function Learning System..."

# Make Python scripts executable
chmod +x collect_data.py
chmod +x create_dataset.py
chmod +x train.py
chmod +x inference.py

echo "✓ Made Python scripts executable"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "✓ Dependencies installed"

# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "Setup complete! 🎉"
echo ""
echo "Quick start:"
echo "1. Collect data:     ./collect_data.py --output-dir ./data"
echo "2. Create dataset:   ./create_dataset.py --data-dir ./data --output-dir ./dataset"  
echo "3. Train model:      ./train.py --dataset-path ./dataset --output-dir ./models"
echo "4. Run inference:    ./inference.py --model-path ./models/best_model.pt"
echo ""
echo "See README.md for detailed instructions!" 