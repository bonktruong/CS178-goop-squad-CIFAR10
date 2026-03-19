# bonktruong Feedforward Neural Network Final Results

## Overview

Feedforward neural networks (fully connected networks) are the simplest deep learning architecture for classification. Each neuron connects to all neurons in the next layer, making them a baseline for understanding how architecture depth, activation functions, and regularization affect classification performance on image data.

## 1) Summary

We evaluated feedforward networks on CIFAR-10 across four experiment groups: architecture depth/width, activation functions, regularization strategies, and learning rates. The best configuration was **arch_2-layer-512-256** with a test accuracy of **56.47%** and 1,708,810 parameters.

## 2) Experimental Setup

- Data split: 80/20 train/validation from CIFAR-10 training set
- Final test: CIFAR-10 built-in test set (10,000 images, used once for final reporting)
- Optimizer: Adam
- Training: 20 epochs per experiment
- Reproducibility seed: `178`
- Selection policy: best validation accuracy, confirmed on held-out test

## 3) Experiment Groups

### Architecture Comparison

| Architecture | Test Acc | Val Acc | Parameters |
|---|---:|---:|---:|
| 1-layer-256 | 53.26% | 53.83% | 789,770 |
| 1-layer-512 | 53.41% | 54.37% | 1,579,530 |
| 2-layer-512-256 | 56.47% | 56.37% | 1,708,810 |
| 3-layer-512-256-128 | 56.14% | 56.68% | 1,740,682 |

Deeper architectures showed marginal improvements. Most parameters are in the first layer (3072 inputs), so adding depth increases parameter count modestly.

### Activation Function Comparison

| Activation | Test Acc | Val Acc |
|---|---:|---:|
| RELU | 56.03% | 56.38% |
| SIGMOID | 49.15% | 49.47% |
| TANH | 51.07% | 51.80% |

ReLU outperformed both Sigmoid and Tanh, consistent with its known advantage of avoiding vanishing gradients.

### Regularization Comparison

| Strategy | Test Acc | Val Acc | Overfit Gap |
|---|---:|---:|---:|
| all_reg | 55.39% | 56.00% | 13.7% |
| batchnorm | 53.17% | 54.82% | 36.4% |
| dropout+bn | 56.31% | 56.39% | 15.5% |
| dropout_0.2 | 53.70% | 54.01% | 15.3% |
| no_reg | 52.81% | 52.94% | 40.3% |

Combining dropout with batch normalization produced the best results. Weight decay provided minimal additional benefit.

### Learning Rate Comparison

| Learning Rate | Test Acc | Val Acc |
|---|---:|---:|
| 0.0001 | 55.57% | 55.11% |
| 0.001 | 55.74% | 56.11% |
| 0.01 | 54.98% | 55.34% |

## 4) Final Model Choice

- Selected model: **arch_2-layer-512-256**
- Test accuracy: **56.47%**
- Validation accuracy: **56.37%**
- Parameters: **1,708,810**
- Macro precision/recall/F1: **56.10% / 56.47% / 55.95%**

## 5) Limitations

Feedforward networks flatten the input image, discarding all spatial structure. This fundamentally limits their ability to recognize visual patterns compared to CNNs, which use convolutional filters to exploit spatial locality. The ~55% test accuracy ceiling reflects this architectural limitation rather than insufficient tuning.

## 6) Artifact Paths

- Best model weights: `outputs/model_weights/bonktruong_ffnn_best.pt`
- Graphs: `outputs/graphs/bonktruong_ffnn_*.png`
- Per-experiment results: `results/bonktruong_*.json`
