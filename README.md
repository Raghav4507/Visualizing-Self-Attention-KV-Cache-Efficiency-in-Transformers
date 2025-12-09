# Visualizing-Self-Attention-KV-Cache-Efficiency-in-Transformers

This project is an interactive **Streamlit** application that demonstrates how Transformer models work internally and why **KV-cache** significantly speeds up autoregressive text generation. It uses **DistilGPT-2** to visualize self-attention patterns and benchmark generation performance with and without caching.

## Features

- Self-Attention visualizer with layer- and head-level heatmaps  
- KV-cache vs no-cache text generation benchmark  
- Per-token timing and speedup comparison  
- Educational focus on Transformer internals and inference optimization  

## Tech Stack

- Python  
- PyTorch  
- Hugging Face Transformers  
- Streamlit  
- Matplotlib, Seaborn, NumPy  

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
Run the App
bash
Copy code
streamlit run app.py

Usage
Use Self-Attention Visualizer to explore attention patterns across layers and heads

Use KV-Cache Benchmark to compare generation speed with and without KV-cache

Check About section for theoretical background and references

Why KV-Cache Matters
During autoregressive generation, KV-cache avoids recomputing attention for the full prefix at every step by reusing past key/value states. This turns quadratic per-token computation into near-linear cost, enabling efficient inference in modern large language models.

References
Vaswani et al., Attention Is All You Need

Radford et al., GPT-2

Hugging Face Transformers
