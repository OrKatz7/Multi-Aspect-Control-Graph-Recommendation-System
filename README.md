# Multi-Aspect-Control-Graph-Recommendation-System

This repository contains the implementation of BGU-GCN, an enhanced Graph Convolutional Network (GCN) architecture for collaborative filtering in recommender systems. BGU-GCN introduces several key optimizations to improve recommendation accuracy and training efficiency compared to state-of-the-art methods.

## Features

- Weighted loss function to prioritize informative training samples
- Adaptive margins to learn fine-grained user-item relationships
- Similarity-based embedding initialization for faster convergence
- Strategic edge removal techniques (bridge removal and noisy edge removal) to reduce computational complexity and improve embedding quality
- Comprehensive evaluation on benchmark datasets (Amazon-Book and Gowalla)

## Requirements

- Python 3.x
- PyTorch
- NumPy
- SciPy
- scikit-learn

## Usage

1. Clone the repository:

```bash
git clone https://github.com/OrKatz7/BGUGCN.git
cd BGUGCN
```

2. Prepare the dataset:
- Download the Amazon-Book and Gowalla datasets
- Preprocess the data and save it in the appropriate format, use preprocessing.py to initilaize the similiarty-based embedding matrix.

3. Train the BGU-GCN model:
```bash
python src/run.py --config_file ./config/ultragcn_gowalla_m1.ini
```

## License

This project is licensed under the [MIT License](LICENSE).


