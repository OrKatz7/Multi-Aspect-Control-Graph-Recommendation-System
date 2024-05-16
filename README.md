# Multi-Aspect-Control-Graph-Recommendation-System

Graph neural networks (GNNs) have recently shown impressive re-
sults in recommender systems, achieving remarkable results across
various sectors. However, their deployment in such systems is not
without challenges. One significant issue is the initialization of
random weights, where incorrect starting points can hinder the
learning process, causing slow convergence or getting stuck in
local minima. To address this, we introduce a novel two-phase
method tailored to the unique demands of recommender systems.
Initially, we refine the latent weights by aligning them with be-
havioral pattern similarities, inspired by the collaborative filtering
technique. Subsequently, we implement a new strategy for manag-
ing the number of negative samples and devising an innovative loss
function. These strategic modifications substantially boost GNN
performance, aligning them more closely with the needs of recom-
mender systems. This is evidenced by our method outperforming
existing state-of-the-art methods on two different public datasets.

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
git clone https://github.com/OrKatz7/Multi-Aspect-Control-Graph-Recommendation-System
cd MultiAspectControlGraphRecommendationSystem
```

2. Prepare the dataset:
- Download the Amazon-Book and Gowalla datasets
- Preprocess the data and save it in the appropriate format, use preprocessing.py to initilaize the similiarty-based embedding matrix.

3. Train:
```bash
run_amazon.sh
```

## License

This project is licensed under the [MIT License](LICENSE).


