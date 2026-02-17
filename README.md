# RAG Dynamic Gatekeeper: The L-Method Implementation

## Overview

In high-stakes Retrieval-Augmented Generation (RAG), fixed similarity thresholds are ineffective. This system implements the **L-Method**, a robust alternative to the naive "elbow" criterion.

Instead of looking for a simple "bend" in the score curve, this algorithm fits two separate linear models to the data:

1. **The Signal Line:** Representing the high-relevance documents.
2. **The Noise Line:** Representing the low-relevance background results.

The algorithm identifies the optimal "pivot point" that minimizes the combined Root Mean Squared Error (RMSE) of these two lines.

## Key Features

* **Mathematical Robustness:** Based on the principle of minimizing residuals rather than visual curvature (addressing concerns raised in *arXiv:2212.12189*).
* **Richness Bias:** Automatically tightens the selection window in high-scoring batches to ensure only the most elite documents are passed to the LLM.
* **Adaptive Salvaging:** In low-signal batches, it identifies the natural break where the "least bad" documents separate from pure database noise.

---

## Behavior & Limitations

### The Pivot Point Inclusion

This implementation identifies the **pivot point** (the index where the curve's behavior changes) and includes it in the "Keep" set.

* In a batch of `[0.95, 0.80, 0.79]`, the algorithm identifies `0.80` as the transition point.
* **Result:** `[True, True, False]`
* *Note:* This ensures high recall at the expense of slight noise inclusion.

## Installation

Requires `numpy`.

```bash
pip install numpy

```
