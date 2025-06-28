# PubMed Graph Attention Network with Explainable AI

<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-3C2179?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-FF6B6B?style=for-the-badge&logo=python&logoColor=white)](https://networkx.org/)

*A comprehensive Graph Neural Network implementation with state-of-the-art explainability features for scientific literature classification*

</div>

---

##  Project Information

### Course Details
- **Course**: Explainable AI
- **Instructor**: Dr. Stefan Heindorf
- **Institution**: Paderborn University
- **Semester**: Summer 2025

### Team Members
- **Mohammadparsa Rostamzadehkhameneh** - *Matriculation Number*: 4038848
- **Alireza Rahnama** - *Matriculation Number*: 4082518

---

##  Project Overview

This project implements a **Graph Attention Network (GAT)** for node classification on the PubMed citation network, focusing on diabetes-related scientific literature classification. The implementation features a comprehensive **Explainable AI (XAI) framework** that provides deep insights into model decision-making through attention pattern analysis, feature importance visualization, and multi-perspective explanations.

###  Classification Task
- **Dataset**: PubMed Citation Network (Planetoid)
- **Task**: Node classification for scientific papers
- **Classes**: 
  - Diabetes Mellitus (Class 0)
  - Experimental Diabetes (Class 1)
  - Type 1 Diabetes (Class 2)
- **Graph Structure**: 19,717 nodes, 108,365 edges, 500 TF-IDF features

---

##  Hardware Dependency Notice

**Note**: The results presented in this project were obtained on a Linux-based OS without GPU acceleration. If you run this code on a GPU or different hardware configuration, you may get different results.

### Reported Results:
- **Original Model**: ~72% accuracy
- **Optimized Model**: ~75% accuracy
- **System**: Linux CPU-based training

### Important:
Results may vary significantly when running on GPU or different hardware setups.

---

##  Key Features

###  Advanced GAT Architecture
- **Multi-head attention mechanism** with configurable attention heads
- **Sophisticated regularization**: Dropout, Batch Normalization, Weight Decay
- **Stable training**: Gradient clipping and early stopping mechanisms
- **Adaptive optimization**: Learning rate scheduling with ReduceLROnPlateau

###  Comprehensive Explainability Framework
- **Attention Pattern Analysis**: Visualize attention flow between papers
- **Feature Importance Analysis**: Gradient-based feature attribution methods
- **Multi-head Attention Visualization**: Compare specialization across attention heads
- **Class-specific Analysis**: Understand feature differences between diabetes types
- **Network-level Insights**: Subgraph attention pattern exploration

###  Advanced Visualization Suite
- **Interactive Network Graphs**: NetworkX-based attention flow visualization
- **Multi-perspective Analysis**: 6 different visualization types per analysis
- **Statistical Visualizations**: Distribution analysis and comparative plots
- **Heatmap Representations**: Attention strength matrices across nodes

###  Intelligent Feature Optimization
- **Dataset-wide Analysis**: Importance evaluation across representative samples
- **Automatic Feature Selection**: Cumulative importance-based filtering
- **Performance Optimization**: Maintaining accuracy with reduced dimensionality
- **Comparative Framework**: Before/after performance evaluation

---

##  Installation & Requirements

### Setup (All Platforms)

```bash
# 1. Clone repository
git clone [your-repository-url]
cd pubmed-gat-explainable-ai

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# Linux/Mac: source venv/bin/activate
# Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
```



### Run the Project
```bash
# Execute the complete pipeline
python pubmed_GNN.py
```

---

##  Usage Instructions

### Quick Start (Complete Pipeline)
```python
# Run the entire pipeline with one command
python pubmed_GNN.py
```

This executes the full pipeline:
1.  Data preprocessing and validation
2.  GAT model training with optimization
3.  Comprehensive explainability analysis
4.  Feature importance evaluation
5.  Dataset optimization and retraining
6.  Performance comparison and insights

### Step-by-Step Execution

#### Step 1: Data Preprocessing
```python
from pubmed_GNN import preprocess_pubmed_dataset

# Preprocess the PubMed dataset
processed_data, save_path = preprocess_pubmed_dataset('processed_data.pt')
print(f"Processed data saved to: {save_path}")
```

#### Step 2: Model Training
```python
from pubmed_GNN import run_PubMed_Gat

# Train the GAT model
model_results = run_PubMed_Gat('./data/processed_data/processed_data.pt')
print(f"Model trained with test accuracy: {model_results['final_test_accuracy']:.4f}")
```

#### Step 3: Explainability Analysis
```python
from pubmed_GNN import explain_gat_attention, explain_gat_features

# Attention pattern analysis
attention_results = explain_gat_attention(model_results, node_id=None)

# Feature importance analysis  
feature_results = explain_gat_features(model_results, top_k=20)
```

#### Step 4: Advanced Analysis
```python
from pubmed_GNN import compare_class_features, analyze_dataset_for_optimization

# Class-specific feature comparison
class_results = compare_class_features(model_results, samples_per_class=5)

# Dataset-wide optimization analysis
optimization_analysis = analyze_dataset_for_optimization(model_results, num_samples=100)
```

---

##  Results & Key Insights

### Model Performance
| Metric | Original GAT | Optimized GAT | Improvement |
|--------|--------------|---------------|-------------|
| **Test Accuracy** | 72.1% | 75.8% | +3.7% |
| **Validation Accuracy** | 75.0% | 78.8% | +3.8% |
| **Feature Count** | 500 | 218 | 56.4% reduction |
| **Model Parameters** | 48,777 | 21,705 | 55.5% reduction |
| **Train-Val Gap** | 0.250 | 0.212 | Better generalization |
| **Importance Retained** | 100% | 80% | Minimal loss |
| **Training Epochs** | 34 (early stop) | 87 (early stop) | More stable convergence |

### Explainability Insights

#### Attention Pattern Discoveries
- **Head Specialization**: Different attention heads focus on distinct types of paper relationships
- **Citation Influence**: Strong attention weights on highly-cited diabetes research papers
- **Class-specific Patterns**: Each diabetes type shows unique attention distribution patterns
- **Local vs Global**: Balance between local neighborhood and global graph structure utilization

#### Feature Importance Findings
- **Sparse Feature Landscape**: Only 22 out of 500 features (4.4%) showed significant importance (>0.01)
- **No High-Impact Features**: Zero features exceeded high importance threshold (>0.05)
- **Top Feature Importance**: Maximum importance score of 0.0269 indicates highly distributed information
- **Noise Identification**: 282 features removed (56.4% reduction) with performance improvement
- **Feature Distribution**: Importance evenly spread across remaining features, suggesting ensemble-like behavior

### Optimization Impact
- **Dimensionality Reduction**: 500 → 180 features (56% reduction)
- **Maintained Performance**: 80% importance retention with improved accuracy
- **Computational Efficiency**: Significant reduction in training time and memory usage
- **Generalization**: Improved model robustness through noise reduction

---

##  Technical Architecture

### GAT Model Architecture
```
Input Layer: [500 TF-IDF features]
↓ [Input Dropout: 0.25]
Multi-Head GAT Layer 1: [32 × 3 heads = 96 dimensions]
↓ [Batch Normalization]
↓ [ReLU Activation]
↓ [Dropout: 0.5]
Single-Head GAT Layer 2: [3 output dimensions]
↓ [Softmax for classification]
Output: [3 diabetes classes]
Parameters: 48,777 (Original) → 21,705 (Optimized)
```

### Explainability Pipeline
```
Model Predictions
    ↓
Attention Weight Extraction → Attention Pattern Analysis
    ↓                              ↓
Gradient Computation → Feature Importance → Visualization Suite
    ↓                              ↓
Subgraph Analysis → Network Visualization → Insights Generation
```

### Data Flow Architecture
```
Raw PubMed Data → Preprocessing → Feature Scaling → GAT Training
                     ↓               ↓              ↓
              Validation Split → Edge Processing → Model Evaluation
                     ↓               ↓              ↓
              XAI Analysis ← Feature Selection ← Performance Analysis
```

---

##  Resources & References

### Core Papers & Methods

#### Graph Neural Networks
- **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., Bengio, Y. (2018)**. "Graph Attention Networks." *International Conference on Learning Representations (ICLR)*.
  - *Implementation*: Core GAT architecture, multi-head attention mechanism

#### Dataset & Graph Construction
- **Sen, P., Namata, G., Bilgic, M., Getoor, L., Gallagher, B., Eliassi-Rad, T. (2008)**. "Collective Classification in Network Data." *AI Magazine, 29(3), 93-106*.
  - *Implementation*: PubMed dataset structure, citation network construction

#### Explainable AI & Feature Attribution
- **Adadi, A., Berrada, M. (2018)**. "Peeking Inside the Black-Box: A Survey on Explainable Artificial Intelligence (XAI)." *IEEE Access, 6, 52138-52160*.
  - *Reference*: XAI theoretical foundations and survey of explainability methods

- **Sundararajan, M., Taly, A., Yan, Q. (2017)**. "Axiomatic Attribution for Deep Networks." *International Conference on Machine Learning (ICML), pp. 3319-3328*.
  - *Implementation*: Gradient-based feature importance calculation (Gradient×Input method)

- **Yuan, H., Yu, H., Gui, S., Ji, S. (2023)**. "Explainability in Graph Neural Networks: A Taxonomic Survey." *IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5), 5782-5799*.
  - *Reference*: Comprehensive survey of GNN explainability methods

#### Graph Neural Network Explainability
- **Ying, R., Bourgeois, D., You, J., Zitnik, M., Leskovec, J. (2019)**. "GNNExplainer: Generating Explanations for Graph Neural Networks." *Advances in Neural Information Processing Systems (NeurIPS), pp. 9240-9251*.
  - *Inspiration*: Explainability framework design, attention-based explanations

- **Vig, J., Belinkov, Y. (2019)**. "Analyzing the Structure of Attention in a Transformer Language Model." *Proceedings of the 2019 ACL Workshop BlackboxNLP, pp. 63-76*.
  - *Reference*: Attention analysis and visualization methodologies

#### Feature Selection & Optimization
- **Guyon, I., Elisseeff, A. (2003)**. "An Introduction to Variable and Feature Selection." *Journal of Machine Learning Research, 3, 1157-1182*.
  - *Implementation*: Feature selection methodology and evaluation metrics

### Technical Libraries & Frameworks
- **Fey, M., Lenssen, J.E. (2019)**. "Fast Graph Representation Learning with PyTorch Geometric." *ICLR Workshop on Representation Learning on Graphs and Manifolds*.
  - *Implementation*: Graph data structures, GAT layers, dataset loading

### Medical Data & Applications
- **Johnson, A.E., Pollard, T.J., Shen, L., et al. (2016)**. "MIMIC-III, a freely accessible critical care database." *Scientific Data, 3, 160035*.
  - *Reference*: Medical data processing and healthcare AI applications

---

##  AI Usage Acknowledgment

In accordance with academic transparency requirements, the following sections utilized AI assistance during development:

### Code Implementation
- **Data Preprocessing Pipeline** (AI-assisted): Complex data validation, feature scaling methodology, and error handling patterns
- **Visualization Functions** (AI-assisted): Matplotlib configuration, NetworkX graph layouts, color schemes, and multi-subplot arrangements
- **Statistical Analysis Methods** (AI-assisted): Feature importance aggregation, distribution analysis, and comparative statistics

### GAT Model Optimization
- **Hyperparameter Tuning** (AI-assisted): Suggestions for optimal learning rates, dropout rates, hidden dimensions, and number of attention heads
- **Advanced Training Techniques** (AI-assisted): Implementation of gradient clipping, learning rate scheduling, early stopping, and batch normalization integration
- **Performance Enhancement Methods** (AI-assisted): Suggestions for attention dropout, weight decay optimization, and multi-layer architecture design

### Visualization Design
- **Attention Network Plots** (AI-assisted): NetworkX layout algorithms, edge styling, node positioning, and interactive elements
- **Statistical Visualizations** (AI-assisted): Histogram configurations, heatmap color maps, and subplot arrangements
- **Comparative Analysis Charts** (AI-assisted): Bar chart designs, legend configurations, and annotation placement

### Analysis Framework
- **Feature Selection Logic** (AI-assisted): Cumulative importance calculation and threshold determination
- **Explainability Framework Design** (AI-assisted): Multi-perspective analysis approach and attention pattern interpretation methods

---

##  Project Structure

```
pubmed-gat-explainable-ai/
├── 📄 pubmed_GNN.py                   # Main implementation file
├── 📄 README.md                       # This documentation
├── 📁 data/
│   ├── 📁 PubMed_data/                # Raw dataset directory (auto-created)
│   └── 📁 processed_data/             # Processed datasets
│       ├── processed_data.pt          # Initial processed data
│       └── feature_selected_data.pt   # Feature-optimized dataset
└── 📁 models/
    └── GAT_Model.pt                   # Trained model checkpoint
```

---


<div align="center">


[⬆ Back to Top](#pubmed-graph-attention-network-with-explainable-ai)

</div>
