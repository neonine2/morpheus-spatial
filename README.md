# Morpheus: Generating Therapeutic Strategies using Spatial Omics

## Introduction

Morpheus is an integrated deep learning framework that takes large scale spatial omics profiles of patient tumors, and combines a formulation of T-cell infiltration prediction as a self-supervised machine learning problem with a counterfactual optimization strategy to generate minimal tumor perturbations predicted to boost T-cell infiltration.

![Graphical summary of the Morpheus framework](assets/summary_fig.png)

## Features

- **Self-Supervised Learning**: Utilizes unlabeled spatial omics data to learn predictive models for T-cell infiltration.
- **Counterfactual Reasoning**: Generates minimal perturbations to the tumor environment, hypothesizing potential improvements in T-cell responses.
- **Deep Learning Integration**: Employs advanced neural network architectures tailored for high-dimensional omics data.
- **Scalability**: Designed to handle large datasets typical of spatial omics studies, enabling robust analysis across numerous patient samples.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- PyTorch Lightning 2.2.0 or higher
- CUDA 11.7 or higher (for GPU acceleration)
- Other dependencies listed in `requirements.txt`

### Installation

Run the following in the command line

```bash
pip install morpheus-spatial
```

### Tutorial
See `example_notebook.ipynb` for a complete workflow on using Morpheus to generate therapeutic strategies.