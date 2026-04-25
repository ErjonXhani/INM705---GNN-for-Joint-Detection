# Yoga Pose GNN

A Graph Neural Network (GNN) project for yoga pose classification using human body keypoints extracted from images. The system converts detected body landmarks into graph representations, where joints are treated as nodes and anatomical connections are treated as edges, and then uses GNN models to classify yoga poses.


Raw Images are in the original repo under this link for space we did not include them 
https://github.com/ErjonXhani/INM705---GNN-for-Joint-Detection/tree/main/data/raw/YogaPoses

## Project Overview

This project investigates the use of Graph Neural Networks for yoga pose recognition. Instead of performing classification directly on raw images, the pipeline first extracts human pose keypoints and then builds a skeleton graph for each image. These graphs are used as input to the GNN model.

The project includes:
- keypoint extraction from yoga pose images
- graph construction from body landmarks
- training and evaluation of GNN-based classifiers
- inference on new input images

## Project Structure

```text
INM705---GNN-FOR-JOINT-DETECTION/
├── checkpoints/                    # Saved model checkpoints
├── configs/                        # Configuration files
├── data/
│   ├── processed/                  # Processed graph data
│   ├── raw/                        # Raw yoga pose images
│   ├── splits/                     # Train/validation/test split files
│   └── make_splits.py              # Script for dataset splitting
├── failed_images/                  # Images where keypoint extraction failed
├── keypoints/                      # Extracted keypoint files
├── outputs/                        # Output results and confusion matrices
├── src/
│   ├── datasets/                   # Dataset definitions
│   ├── evaluation/                 # Metrics and evaluation utilities
│   ├── models/                     # GCN, GAT, GIN, GraphSAGE models
│   ├── preprocessing/              # Keypoint extraction and graph conversion
│   ├── training/                   # Training pipeline
│   └── utils/                      # Helper utilities
├── wandb/                          # Experiment tracking logs
├── confusion_matrix_eval.py        # Confusion matrix evaluation script
├── inference.py                    # Inference script
├── train.py                        # Main training script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── CONTRIBUTIONS.md                # Contribution guidelines
└── .gitignore
```
## Requirements

**Recommended environment:**
- Python 3.10 or 3.11
- pip
- virtual environment support

This project was developed in a virtual environment.

## Installation

### 1. Clone the repository

git clone <repository-url>
cd INM705---GNN-FOR-JOINT-DETECTION 

### 2. Create and activate a virtual environment
python -m venv venv

Note: If multiple Python versions are installed, use: py -3.10 -m venv venv

On Mac/ Linux:
source venv/bin/activate

On Windows:

venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

## PyTorch Geometric Note
PyTorch Geometric may require additional environment-specific dependencies on some systems.

If installation issues occur, please refer to the official installation guide:

https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## Data

The raw yoga pose images are located in: data/raw/YogaPoses/

### Dataset Classes

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

### Preprocessing Pipeline

Before training, the dataset should go through the following steps:

1. Extract pose keypoints from the raw yoga images  
2. Convert extracted keypoints into graph representations  
3. Split the dataset into training, validation, and test sets  
4. Save processed graph data in the `data/processed/` directory  

### Visualization of keypoints

python src/preprocessing/visualize.py

## Usage

### Training
Train the model using the yaml configuration:

python train.py --config configs/baseline.yaml

Erjon's WandB key: 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

### Evaluation / Inference / 3 Best Models

Inference in this project is performed as part of the evaluation pipeline on the test dataset.

python inference.py --checkpoint checkpoints/sage_final_no_conf_erjon.pt --config configs/sage_final_no_conf.yaml

python inference.py --checkpoint checkpoints/gat_heads8_hidden256_erjon.pt --config configs/gat_heads_8_hidden_256.yaml

python inference.py --checkpoint checkpoints/gin_final_erjon.pt --config configs/final_gin.yaml

For example, evaluation scripts can be used to generate performance metrics and confusion matrices:
python confusion_matrix_eval.py --checkpoint checkpoints/sage_final_no_conf_erjon.pt --config configs/sage_final_no_conf.yaml --split test


**Note:**
- The model operates on graph-structured data derived from pose keypoints  
- Raw images cannot be used directly for inference  
- Images must first be preprocessed through keypoint extraction and graph conversion before being passed to the model 

### Configuration

Model and training parameters can be modified in: configs/ baseline.yaml

These may include:
- hidden dimension
- number of GNN layers
- dropout
- learning rate
- batch size
- number of epochs

## Outputs
During execution, the project may generate:
- trained model checkpoints in checkpoints/
- output figures (e.g., confusion matrices) in outputs/


## Contributing

See CONTRIBUTIONS.md for contribution guidelines.

## License

This project is licensed under the MIT License.