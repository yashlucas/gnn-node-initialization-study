# GNN Node Initialization Study

## Overview

This repository contains the implementation and experimental results for a Graph Representation Learning course project at TU Wien.

The project investigates how different node initialization strategies affect the performance, robustness, and information propagation behavior of Graph Neural Networks (GNNs). In addition, feature masking and depth-based experiments are conducted to study robustness and oversmoothing effects.

---

## Research Question

**How do node initialization and feature degradation affect information propagation and generalization in Graph Neural Networks?**

This study focuses on understanding:

- The importance of node features in graph learning
- The role of structural node information
- Robustness to missing or degraded features
- The effect of network depth on performance
- Differences between GCN, GIN, and GraphSAGE architectures

---

## Models Evaluated

The following Graph Neural Network architectures are compared:

### Graph Convolutional Network (GCN)
Kipf & Welling (2017)

### Graph Isomorphism Network (GIN)
Xu et al. (2019)

### GraphSAGE
Hamilton et al. (2017)

---

## Dataset

### ENZYMES Dataset

The ENZYMES dataset from the TU benchmark collection is used.

Characteristics:

- 600 graphs
- 6 enzyme classes
- Graph classification task
- Node feature vectors available

---

## Experiments

### 1. Node Initialization Study

We compare the following initialization strategies:

| Initialization | Description |
|--------------|-------------|
| Original | Original dataset node features |
| Constant | All nodes initialized with identical values |
| Random | Random feature vectors |
| Degree | Node degree only |
| Degree + Clustering | Degree and clustering coefficient |

Goal:

- Study the effect of node representations on graph classification performance.

---

### 2. Feature Masking Study

Node features are progressively removed to evaluate robustness.

Masking levels:

- 0%
- 25%
- 50%
- 75%
- 100%

Goal:

- Measure dependence on node attributes.
- Analyze robustness under degraded feature quality.

---

### 3. Oversmoothing Study

The number of message-passing layers is varied:

- 2 Layers
- 3 Layers
- 4 Layers
- 5 Layers

Goal:

- Investigate whether deeper GNNs improve information propagation or suffer from oversmoothing.

---

## Main Findings

### Node Initialization

- Original node features consistently achieve the highest accuracy.
- Structural features (degree and clustering coefficient) outperform constant and random initialization.
- GIN achieves the best overall performance.

### Feature Masking

- Performance decreases as node features are removed.
- Original node attributes contain important task-specific information.
- GraphSAGE exhibits relatively stable behavior under feature degradation.

### Oversmoothing

- Increasing depth does not always improve performance.
- GIN performs best with moderate depth.
- GCN benefits from deeper architectures.
- GraphSAGE performance decreases when additional layers are added.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── Final_Project.ipynb
│
├── figures/
│   ├── initialization_results.png
│   ├── masking_results.png
│   └── oversmoothing_results.png
│
├── results/
│   ├── initialization_results.csv
│   ├── masking_results.csv
│   └── oversmoothing_results.csv
│
└── report/
    └── report.pdf

---

## Installation

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dependencies

Main packages:

```text
torch
torch-geometric
numpy
pandas
matplotlib
networkx
scikit-learn
jupyter
```

---

## Running Experiments

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/gnn_experiments.ipynb
```

and execute the cells sequentially.

---

## References

1. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks.
2. Hamilton, W., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs.
3. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful are Graph Neural Networks?
4. Morris, C. et al. (2020). TUDataset: A Collection of Benchmark Datasets for Learning with Graphs.
5. Li, Q., Han, Z., & Wu, X. (2018). Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning.

---

## Author

**Yash Lucas**  
Department of Informatics  
TU Wien, Vienna, Austria

Course: Graph Representation Learning
