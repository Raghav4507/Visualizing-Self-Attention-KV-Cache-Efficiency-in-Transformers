# Visualizing-Self-Attention-KV-Cache-Efficiency-in-Transformers

An interactive **Streamlit** application for understanding Transformer internals and inference optimization.  
This project visualizes **multi-head self-attention** and benchmarks **KV-cache vs. non-cached** autoregressive generation using **DistilGPT-2**.

---

## Overview

Transformer self-attention has quadratic complexity with respect to sequence length, making inference expensive during autoregressive generation. Modern language models solve this using **Key–Value (KV) caching**, which reuses previously computed attention states.

This project provides a clear, hands-on demonstration of both concepts through visualization and benchmarking.

---

## Features

- Self-attention heatmaps for each layer and head  
- Token-level attention visualization  
- Autoregressive text generation benchmark  
- Performance comparison with and without KV-cache  
- Educational focus on Transformer internals  

---

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
---

## Running the App

```bash
streamlit run app.py
```
---

## Usage

### Self-Attention Visualizer
Explore how attention is distributed across tokens for different layers and heads.

### KV-Cache Benchmark
Compare generation time per token with and without KV-cache to observe inference speedups.

### About Section
Provides theoretical background and learning references.

---

## Why KV-Cache Matters

In naive autoregressive decoding, the model recomputes attention over the entire sequence for every new token, leading to quadratic per-token computation. KV-caching stores past key/value tensors and reuses them during generation, reducing computation to near-linear complexity and enabling efficient inference in large language models.

---

## Model

DistilGPT-2
Loaded with attention outputs enabled for visualization and caching support for benchmarking.

---

## References

- Vaswani et al., *Attention Is All You Need*  
- Radford et al., *Language Models are Unsupervised Multitask Learners (GPT-2)*  
- Hugging Face Transformers Documentation


