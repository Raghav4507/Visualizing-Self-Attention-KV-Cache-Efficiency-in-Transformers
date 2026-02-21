import streamlit as st
import torch
from model import (
    get_tokenizer,
    get_attentions_for_text,
    generate_no_cache,
    generate_with_cache,
)



st.set_page_config(page_title="Visualizing-Self-Attention-KV-Cache-Efficiency-in-Transformers Demo", layout="wide")

st.title("💡 Visualizing-Self-Attention-KV-Cache-Efficiency-in-Transformers")

# Sidebar navigation
mode = st.sidebar.radio("Select Mode", ("Self-Attention Visualizer", "KV-Cache Benchmark", "About"))

tokenizer = get_tokenizer()

if mode == "Self-Attention Visualizer":
    st.header("Self-Attention Visualizer")
    prompt = st.text_area("Enter prompt text:", value="The quick brown fox jumps over the lazy dog.")
    seq_len = st.slider("Max length (for attention):", min_value=8, max_value=256, value=32, step=8)
    if st.button("Compute Attention"):
        with st.spinner("Computing attention..."):
            data = get_attentions_for_text(prompt, max_len=seq_len)
            tokens = data["tokens"]
            attn = data["attentions"]  # tuple of (batch, heads, seq, seq)
            # Convert to stacked tensor (layers, heads, seq, seq)
            stacked = torch.stack(attn, dim=0).squeeze(1).detach().cpu()  # shape L x h x seq x seq

            # Slider for layer and head selection
            num_layers, num_heads, seq_len_, _ = stacked.shape
            layer_idx = st.slider("Select Layer", min_value=0, max_value=num_layers-1, value=0)
            head_idx = st.slider("Select Head", min_value=0, max_value=num_heads-1, value=0)

            # Get specific attention matrix
            attn_matrix = stacked[layer_idx, head_idx, :, :]

            # Plot heatmap
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np

            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(attn_matrix.numpy(), xticklabels=tokens, yticklabels=tokens, cmap="viridis", ax=ax)
            ax.set_title(f"Attention Heatmap - Layer {layer_idx}, Head {head_idx}")
            ax.set_xlabel("Key Tokens")
            ax.set_ylabel("Query Tokens")
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            st.pyplot(fig)

            st.markdown("**Tokens:**")
            st.write(tokens)
            st.markdown("**Notes:** Heatmap shows attention from query (row) to key (column).")


elif mode == "KV-Cache Benchmark":
    st.header("KV-Cache Generation Benchmark")
    prompt = st.text_area("Enter prompt for generation:", value="Once upon a time")
    gen_len = st.slider("Number of tokens to generate:", min_value=1, max_value=64, value=16)
    run_bench = st.button("Run Benchmark")
    if run_bench:
        st.write("Running naive (no-cache) generation …")
        no = generate_no_cache(prompt, gen_len=gen_len)
        st.write("Running cached generation …")
        ca = generate_with_cache(prompt, gen_len=gen_len)

        t_no = sum(no["times_per_step_s"])
        t_ca = sum(ca["times_per_step_s"])

        st.subheader("Results")
        st.write(f"No-cache total time: **{t_no:.4f} s**")
        st.write(f"With-cache total time: **{t_ca:.4f} s**")
        speedup = t_no / t_ca if t_ca > 0 else float("inf")
        st.write(f"Speedup (no-cache ÷ cache): **{speedup:.2f}×**")
        st.markdown("**Generated text (no-cache):**")
        st.code(no["generated_text"])
        st.markdown("**Generated text (with-cache):**")
        st.code(ca["generated_text"])

        # Plot per-step times
        import numpy as np
        import matplotlib.pyplot as plt

        times_no = no["times_per_step_s"]
        times_ca = ca["times_per_step_s"]
        x = np.arange(len(times_no))
        fig, ax = plt.subplots()
        ax.plot(x, times_no, label="No-cache")
        ax.plot(x, times_ca, label="With-cache")
        ax.set_xlabel("Generation Step")
        ax.set_ylabel("Time (s)")
        ax.legend()
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("**Insight:** With KV-cache, the model reuses its past key/value states, so each step only processes the new token — reducing per-step cost compared to reprocessing the entire prefix.")

elif mode == "About":
    st.header("About This Project")
    st.markdown(
        """
        This application is built for educational and algorithm‑analysis purposes.  
        It demonstrates:
        - **Self‑Attention**: how each layer and head attends to tokens in a Transformer model  
        - **KV‑Cache Optimization**: how autoregressive generation can be accelerated by reusing past key/value states (reducing computation)  

        **Why it matters for algorithms:**  
        - Attention has **quadratic complexity** w.r.t. sequence length  
        - KV‑Cache reduces computation from O(n²) per token to roughly O(n) per token during generation  
        - These algorithmic improvements power modern LLMs like GPT  

        ---
        **References you can use in your report:**  
        1. Vaswani et al. (2017), *“Attention Is All You Need”*  
        2. Radford et al., *GPT-2 Paper*  
        3. HuggingFace Transformers Library  
        4. Various works on caching / inference optimization
        """
    )
