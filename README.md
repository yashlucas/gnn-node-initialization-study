# GNN Node Initialization Study

## Overview

This repository contains comprehensive implementations and experimental analysis for a Graph Representation Learning course project at TU Wien. The project investigates how different node initialization strategies affect the performance, robustness, and information propagation behavior of Graph Neural Networks (GNNs).

The study combines classical initialization strategies with advanced structural encodings, feature robustness analysis, and deep network diagnostics to provide actionable insights for GNN practitioners.

---

## Research Question

**How do node initialization strategies, feature degradation, and network depth affect information propagation and generalization in Graph Neural Networks?**

This comprehensive study addresses:

- **Node Initialization Impact**: How different initialization strategies (from simple to advanced) affect GNN performance
- **Structural Information Encoding**: The role of graph topology in feature representation
- **Feature Robustness**: Model resilience under missing or degraded node features
- **Network Depth Analysis**: The relationship between model depth, oversmoothing, and performance
- **Architectural Differences**: Comparative analysis of GCN, GIN, and GraphSAGE across all conditions

---

## Models Evaluated

Three state-of-the-art Graph Neural Network architectures are comprehensively evaluated:

### 1. Graph Convolutional Network (GCN)
**Reference**: Kipf & Welling (2017) — Semi-Supervised Classification with GCNs
- Spectral-inspired, spatially-localized filters
- Efficient 1-hop aggregation with normalization

### 2. Graph Isomorphism Network (GIN)
**Reference**: Xu et al. (2019) — How Powerful are Graph Neural Networks?
- Provably as expressive as Weisfeiler-Lehman test
- Non-linear multi-layer perceptron aggregation

### 3. GraphSAGE
**Reference**: Hamilton et al. (2017) — Inductive Representation Learning on Large Graphs
- Sampling-based inductive learning
- Flexible neighborhood aggregation functions

---

## Dataset

### ENZYMES (TU Benchmark Collection)

**Source**: Morris, C. et al. (2020) — TUDataset: A Collection of Benchmark Datasets for Learning with Graphs

**Characteristics**:
- **Size**: 600 protein graphs
- **Task**: Binary enzyme classification (6 classes)
- **Graph Properties**: Avg nodes = 32.6, Avg edges = 62.1
- **Node Features**: Available for all nodes
- **Challenge**: Small graphs with high class imbalance

---

## Experiments

### Experiment 1: Node Initialization Study

**Basic Strategies**:

| Initialization | Dimensionality | Description |
|----------------|----------------|-------------|
| **Original** | 18 | Original dataset node features |
| **Constant** | 1 | All nodes identical (constant 1.0) |
| **Random** | 18 | Random Gaussian initialization |
| **Degree** | 1 | Node degree normalized |
| **Degree + Clustering** | 2 | Degree + clustering coefficient, standardized |

**Advanced Strategies**:

| Initialization | Dimensionality | Description |
|----------------|----------------|-------------|
| **Centrality Bundle** | 6 | Degree, clustering, PageRank, betweenness, closeness, core number |
| **Degree Bins** | 6 | One-hot encoded degree bins |
| **Spectral** | 4 | Laplacian eigenvector positional encodings |
| **Structure + Spectral** | 10 | Centrality bundle + spectral features combined |
| **Weisfeiler-Lehman Roles** | 32 | WL-based structural role one-hot encoding (2 iterations) |
| **Original + Structure** | 24 | Original features concatenated with centrality bundle |

**Goals**:
- Quantify the effect of node initialization quality on graph classification
- Compare simple structural proxies vs. learned original features
- Identify optimal initialization for each GNN architecture
- Validate advanced initializations with k-fold cross-validation

---

### Experiment 2: Feature Masking (Robustness Study)

**Masking Ratios**: 0%, 25%, 50%, 75%, 100%

**Methodology**:
- Progressively mask (zero out) node feature dimensions
- Evaluate model robustness under information degradation
- Measure graceful performance degradation curves

**Goals**:
- Assess dependence on node attributes
- Compare robustness across GNN architectures
- Identify minimum feature quality thresholds
- Analyze trade-offs between feature importance and model generalization

---

### Experiment 3: Network Depth & Oversmoothing Analysis

**Layer Configurations**: 2, 3, 4, 5, 6, 8 layers

**Methodology**:
- Train GNNs with varying depths using optimal initialization
- Measure test accuracy and validation F1 scores
- Track pairwise cosine similarity across layers as oversmoothing diagnostic
- Analyze layer-wise embedding convergence

**Goals**:
- Investigate the "depth paradox" in message-passing networks
- Detect oversmoothing effects (when node embeddings become indistinguishable)
- Identify optimal network depth for each architecture
- Provide empirical evidence on depth-performance relationship

---

## Main Findings

### Node Initialization

 **Original node features** consistently achieve highest accuracy across all models
 **Degree + Clustering** is the best **structural-only initialization** (3-5% above random)
 **Centrality Bundle** provides competitive performance with **6 diverse structural metrics**
 **GIN performs best overall** (mean accuracy 0.5847) across initialization strategies
 **GraphSAGE shows stability** even with degraded or minimal initialization

**Key Insight**: Advanced structural initializations can replace original features when unavailable, with **centrality_bundle** being the most practical choice for scalability.

---

### Feature Masking (Robustness)

 **Performance degrades gracefully** from 0% to 50% masking across all models
 **Steep drop at 75% masking** indicates critical feature threshold
 **GraphSAGE most robust** (retention: 92% at 50% masking)
 **GCN most sensitive** to feature degradation
 **GIN balanced** with moderate robustness (89% retention at 50% masking)

**Key Insight**: Models require ~40-50% of original features for reasonable performance; beyond that, architecture-specific robustness varies significantly.

---

### Oversmoothing Effects

 **GIN performs best at moderate depth** (3 layers: 0.5847 accuracy)
 **GCN benefits from deeper architectures** (5 layers: 0.5120 vs. 3 layers: 0.4998)
 **GraphSAGE degrades with depth** (5 layers: 0.4687 vs. 3 layers: 0.5073)
 **Cosine similarity increases with depth** indicating oversmoothing onset
 **Optimal depth differs by architecture**: GIN (3), GCN (5), GraphSAGE (3-4)

**Key Insight**: Oversmoothing is architecture-dependent. GIN's expressive aggregation mitigates depth issues, while GraphSAGE's sampling may amplify them.

---

## Repository Structure

```text
.
├── GNN.py                              # Main implementation (production-ready, 1200+ lines)
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
│
├── results/                            # Baseline experimental results
│   ├── 01_initialization.csv           # Initialization study (mean/std across seeds)
│   ├── 02_masking.csv                  # Feature masking results (5 ratios × 3 models)
│   ├── 03_oversmoothing.csv            # Network depth analysis (4 depths × 3 models)
│   └── visualizations/
│       ├── 01_initialization_comparison.png
│       ├── 02_masking_robustness.png
│       ├── 03_oversmoothing_effect.png
│       ├── 04_model_ranking.png
│       ├── 05_initialization_heatmap.png
│       └── 06_comprehensive_dashboard.png
│
└── results_complete_research/          # Intensive study outputs (generated on demand)
    ├── research_01_kfold_cv.csv        # 5-fold CV results (5 inits × 3 models × 5 folds = 75 rows)
    ├── research_02_hyperparameter_grid.csv  # Grid search results (48 configurations)
    ├── research_03_statistical_tests.csv    # Pairwise statistical comparisons
    ├── research_04_runtime_profile.csv      # Runtime profiling of all methods
    └── visualizations/
        ├── 07_cross_validation_analysis.png
        ├── 08_fold_performance.png
        ├── 09_hyperparameter_sensitivity.png
        ├── 10_runtime_profile.png
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Step 1: Create Virtual Environment

```bash
python -m venv gnn_env
source gnn_env/bin/activate  # On Windows: gnn_env\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import torch; import torch_geometric; print('Installation successful')"
```

---

## Dependencies

### Core Libraries

```text
torch>=2.0.0           # Deep learning framework
torch-geometric>=2.3.0 # Graph neural networks
```

### Scientific Computing

```text
numpy>=1.21.0          # Numerical operations
pandas>=1.3.0          # Data manipulation & analysis
scipy>=1.7.0           # Statistical functions
scikit-learn>=1.0.0    # Machine learning utilities
```

### Visualization & Analysis

```text
matplotlib>=3.4.0      # Plotting
seaborn>=0.11.0        # Statistical visualizations
```

### Graph Processing

```text
networkx>=2.6.0        # Graph algorithms & metrics
```

---

## Running Experiments

### Quick Start: Baseline Experiments (5-10 minutes)

```bash
python GNN.py
```

This runs all three baseline experiments and generates results in `results/`:
- Node initialization study (3 models × 5 strategies)
- Feature masking study (3 models × 5 ratios)
- Network depth study (3 models × 4 depths)

**Outputs**: 
- CSV files: `01_initialization.csv`, `02_masking.csv`, `03_oversmoothing.csv`
- 6 comprehensive visualizations

---

### Advance Analysis
For rigorous k-fold cross-validation, hyperparameter search, and statistical testing:

```python
from pathlib import Path
from GNN import run_full_intensive_study, dataset

# Run the full intensive study
results = run_full_intensive_study(dataset, results_dir_gnn=Path('results_complete_research'))
```

**Outputs**: 
- 5-fold cross-validation across 5 initializations × 3 models
- 48-configuration hyperparameter grid search
- Paired statistical significance tests
- Runtime profiling for all initialization methods
- 5 advanced analysis visualizations

**Results Directory**: `results_complete_research/`

---

## Core Functions & Utilities

### 1. Node Initialization

```python
from GNN import add_node_initialization, add_advanced_node_initialization, initialize_graphs_with_structure

# Basic strategies
data = add_node_initialization(dataset, init_type='degree_clustering', seed=42)

# Advanced strategies (recommended)
data = add_advanced_node_initialization(dataset, init_type='centrality_bundle', seed=42)
data = add_advanced_node_initialization(dataset, init_type='structure_plus_spectral', seed=42)
data = add_advanced_node_initialization(dataset, init_type='wl_roles', seed=42)

# Production-ready function
data = initialize_graphs_with_structure(dataset, init_type='degree_clustering')
```

### 2. Model Architectures

Three implementations provided with batch normalization and dropout:

```python
from GNN import BetterGCN, BetterGIN, BetterGraphSAGE

model = BetterGCN(input_dim=18, hidden_dim=64, num_classes=6, 
                  num_layers=3, dropout=0.5)
model = BetterGIN(input_dim=18, hidden_dim=64, num_classes=6, 
                  num_layers=3, dropout=0.5)
model = BetterGraphSAGE(input_dim=18, hidden_dim=64, num_classes=6, 
                        num_layers=3, dropout=0.5)
```

### 3. Training & Evaluation

```python
from GNN import train_one_epoch, evaluate, evaluate_detailed, run_experiment, run_experiment_detailed

# Basic training
train_one_epoch(model, loader, optimizer, device)
acc = evaluate(model, loader, device)

# Detailed metrics (accuracy, F1, precision, recall, confusion matrix)
metrics = evaluate_detailed(model, loader, device)

# Single experiment run
results = run_experiment(dataset, BetterGCN, model_name='GCN', 
                        setting_name='test', epochs=50)

# Enhanced training with early stopping & learning rate scheduling
metrics = run_experiment_detailed(dataset, BetterGCN, model_name='GCN', 
                                 setting_name='test', epochs=200, patience=30)
```

### 4. Cross-Validation & Hyperparameter Search

```python
from GNN import run_kfold_cv, run_hyperparameter_grid

# 5-fold stratified cross-validation
cv_df = run_kfold_cv(dataset, BetterGCN, model_name='GCN', 
                     setting_name='degree_clustering', k=5, epochs=50)

# Hyperparameter grid search (48 configurations default)
grid_df = run_hyperparameter_grid(
    dataset, 
    model_class=BetterGCN,
    model_name='GCN',
    init_types=('original', 'degree_clustering', 'centrality_bundle'),
    hidden_dims=(32, 64),
    dropouts=(0.3, 0.5),
    learning_rates=(0.001, 0.0005),
    layers=(2, 3, 4),
    epochs=50
)
```

### 5. Diagnostics & Analysis

```python
from GNN import oversmoothing_profile, compare_settings_paired, profile_initialization_runtime

# Oversmoothing diagnostic (cosine similarity by layer)
smoothing_df = oversmoothing_profile(dataset, num_layers_list=[2,3,4,5,6,8])

# Statistical significance testing (paired t-test, Wilcoxon, Cohen's d)
stats_df = compare_settings_paired(cv_results, 
                                  setting_col='initialization',
                                  metric_col='macro_f1',
                                  pair_col='fold')

# Runtime profiling
runtime_df = profile_initialization_runtime(
    dataset, 
    init_types=['original', 'degree', 'degree_clustering', 'centrality_bundle'],
    repeats=3
)
```

---

## Experiment Results Summary

### Initialization Study Results (Mean Accuracy ± Std Dev)

| Model | Original | Constant | Random | Degree | Degree+Clust |
|-------|----------|----------|--------|--------|--------------|
| **GCN** | 0.5143 ± 0.0142 | 0.3571 | 0.4571 | 0.5143 | 0.5000 |
| **GIN** | 0.5857 ± 0.0142 | 0.3714 | 0.4286 | 0.5571 | 0.5429 |
| **GraphSAGE** | 0.5214 ± 0.0142 | 0.3571 | 0.4571 | 0.4857 | 0.4857 |

**Best Configuration**: GIN + Original (0.5857 accuracy)
**Best Structural-Only**: GIN + Degree (0.5571 accuracy)
**Best Advanced Init**: GIN + Centrality Bundle (0.56 accuracy, highly stable)

---

### Feature Masking Results

| Model | 0% Masked | 25% Masked | 50% Masked | 75% Masked | 100% Masked |
|-------|-----------|-----------|-----------|-----------|------------|
| **GCN** | 0.5143 | 0.5000 | 0.4286 | 0.3571 | 0.2857 |
| **GIN** | 0.5857 | 0.5571 | 0.5286 | 0.3143 | 0.2286 |
| **GraphSAGE** | 0.5214 | 0.5143 | 0.4857 | 0.3857 | 0.2571 |

**Key Threshold**: Performance retention at 50% masking averages 88-92%
**Most Robust Model**: GraphSAGE (92% retention at 50% masking)

---

### Network Depth Results

| Model | 2 Layers | 3 Layers | 4 Layers | 5 Layers |
|-------|----------|----------|----------|----------|
| **GCN** | 0.4856 | 0.4998 | 0.5069 | 0.5120 |
| **GIN** | 0.5712 | 0.5847 | 0.5612 | 0.5473 |
| **GraphSAGE** | 0.5165 | 0.5073 | 0.4932 | 0.4687 |

**Optimal Depths**: GIN (3 layers), GCN (5 layers), GraphSAGE (2-3 layers)
**Oversmoothing Onset**: GIN & GraphSAGE show degradation beyond optimal depth; GCN benefits from depth up to 5

---

## Visualization Guide

### Baseline Results (6 visualizations)

1. **01_initialization_comparison.png**: Line plot with error bars + box plot distribution
2. **02_masking_robustness.png**: Robustness curves across masking ratios
3. **03_oversmoothing_effect.png**: Accuracy vs. network depth by architecture
4. **04_model_ranking.png**: Bar chart with error bars across initializations
5. **05_initialization_heatmap.png**: Heatmap of model × initialization performance
6. **06_comprehensive_dashboard.png**: 3×3 multi-panel summary dashboard

### Advanced Results (5 visualizations)

7. **07_cross_validation_analysis.png**: Box plots and bar charts of 5-fold CV results
8. **08_fold_performance.png**: Fold-wise accuracy trends by initialization
9. **09_hyperparameter_sensitivity.png**: 2×2 grid of hyperparameter sensitivity curves
10. **10_runtime_profile.png**: Computational cost comparison (seconds)


---

## Implementation Details

### Advanced Initialization Methods

#### Centrality Bundle
Combines 6 graph-theoretic measures:
- Node degree
- Clustering coefficient
- PageRank centrality
- Betweenness centrality
- Closeness centrality
- Core number

**Advantages**: Captures diverse structural roles; proven to work across graph types
**Computational Cost**: O(n + m) for degree; O(n² + m) for centrality measures

#### Spectral Positional Encoding
Uses Laplacian eigenvectors as node features:
- Normalized Laplacian: L = I - D⁻¹/² A D⁻¹/²
- Extract top-k eigenvectors (excluding trivial first eigenvector)
- Captures global graph structure

**Advantages**: Theoretically grounded in spectral graph theory
**Computational Cost**: O(n³) for eigendecomposition

#### Weisfeiler-Lehman Structural Roles
Approximates WL graph isomorphism test with structural roles:
- 2-iteration WL labeling of nodes
- One-hot encode resulting role assignments
- Captures local-to-regional structure

**Advantages**: Provably captures graph structure up to WL expressiveness
**Computational Cost**: O(n × iterations × avg_degree)

---

## Key Hyperparameters

```python
# GNN Architecture
hidden_dim = 64         # Hidden layer dimension (32, 64, 128 tested)
num_layers = 3          # Network depth (2-6 range; 3 recommended)
dropout = 0.5           # Dropout rate (0.3-0.7 recommended)

# Training
learning_rate = 0.001   # Adam optimizer learning rate
weight_decay = 1e-4     # L2 regularization coefficient
batch_size = 16         # Batch size (16-32 for small graphs)
epochs = 50-200         # Training epochs (50 for quick runs, 200 for intensive)
patience = 30           # Early stopping patience

# Optimization
optimizer = 'Adam'      # SGD, Adam, AdamW all viable
scheduler = 'ReduceLROnPlateau'  # Learning rate decay strategy
```

**Recommended Settings**:
- **Quick Prototyping**: hidden_dim=32, epochs=50, num_layers=2
- **Production**: hidden_dim=64, epochs=200, num_layers=3, patience=30
- **Research**: hidden_dim=64, epochs=200, num_layers=3-5, with grid search

---

## Extending the Framework

### Adding a New Initialization Strategy

```python
from GNN import set_seed, add_node_initialization
import torch
import copy

def add_custom_initialization(dataset, init_type='custom', seed=42):
    set_seed(seed)
    new_dataset = []
    for graph in dataset:
        g = copy.deepcopy(graph)
        n = g.num_nodes
        
        # Your initialization logic here
        g.x = torch.randn((n, 16))  # Replace with your method
        
        new_dataset.append(g)
    return new_dataset

# Use it
custom_dataset = add_custom_initialization(dataset)
from GNN import run_experiment, BetterGCN
results = run_experiment(custom_dataset, BetterGCN, 'GCN', 'custom', epochs=50)
```

### Adding a New GNN Architecture

```python
import torch
from torch.nn import Linear, ModuleList, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class CustomGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = ModuleList([GCNConv(input_dim, hidden_dim)])
        self.bns = ModuleList([BatchNorm1d(hidden_dim)])
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm1d(hidden_dim))
        self.lin = Linear(hidden_dim, num_classes)
        self.dropout_rate = dropout
        
    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = global_mean_pool(x, batch)
        return self.lin(x)

# Use it
from GNN import run_experiment
results = run_experiment(dataset, CustomGNN, 'CustomGNN', 'test_setting', epochs=50)
```

---

## Code Statistics

| Metric | Value |
|--------|-------|
| **Total Lines** | 1200+ |
| **Initialization Methods** | 11 (5 basic + 6 advanced) |
| **GNN Architectures** | 3 (GCN, GIN, GraphSAGE) |
| **Experiments** | 3 (Initialization, Masking, Depth) |
| **Training Utilities** | 10+ functions |
| **Evaluation Metrics** | 6+ (accuracy, F1, precision, recall, etc.) |
| **Visualization Functions** | 6 baseline + 5 advanced = 11 total |
| **Utility Functions** | 20+ |

---
