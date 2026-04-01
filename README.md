# Yoga Pose GNN

A Graph Neural Network (GNN) model for yoga pose classification and analysis using keypoint detection.

## Project Structure

```
yoga-pose-gnn/
├── data/                    # Data directory
│   ├── raw/                 # Original yoga pose images
│   ├── processed/           # Extracted keypoints and graphs
│   └── splits/              # Train/val/test split files
├── notebooks/               # Jupyter notebooks
│   └── openpose_testing.ipynb
├── src/                     # Source code
│   ├── datasets/            # Dataset classes
│   ├── models/              # Model architectures
│   ├── preprocessing/       # Data preprocessing scripts
│   ├── training/            # Training utilities
│   ├── evaluation/          # Evaluation metrics
│   └── utils/               # Helper functions
├── checkpoints/             # Saved model checkpoints
├── outputs/                 # Output results
│   ├── figures/             # Visualization figures
│   └── predictions/         # Model predictions
├── configs/                 # Configuration files
├── train.py                 # Training script
├── inference.py             # Inference script
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── .gitignore               # Git ignore rules
└── CONTRIBUTIONS.md         # Contribution guidelines
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd yoga-pose-gnn
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the model using the baseline configuration:
```bash
python train.py --config configs/baseline.yaml
```

### Inference

Run inference on a single image:
```bash
python inference.py --checkpoint checkpoints/model.pth --image path/to/image.jpg
```

## Data

The raw yoga pose images are located in `data/raw/YogaPoses/`. The dataset includes the following poses:
- Camel
- Cobra
- Cow
- Crane
- Downdog
- Eagle
- Fish
- Goddess
- Lotus
- Plank
- Revolved Triangle
- Tree
- Warrior
- Wheel

## Configuration

Model and training parameters can be modified in `configs/baseline.yaml`.

## Contributing

See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for guidelines on contributing to this project.

## License

This project is licensed under the MIT License.
