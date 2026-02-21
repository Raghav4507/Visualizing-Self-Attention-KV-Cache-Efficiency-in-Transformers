import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "distilgpt2"

_tokenizer = None
_model = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer

def get_model():
    global _model
    if _model is None:
        # Load model with attentions and cache support
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_attentions=True, use_cache=True)
        _model.to(DEVICE)
        _model.eval()
    return _model

def get_attentions_for_text(text: str, max_len: int = 64) -> Dict:
    tokenizer = get_tokenizer()
    model = get_model()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = enc["input_ids"].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    attentions = outputs.attentions  # tuple of tensors
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return {
        "tokens": tokens,
        "input_ids": input_ids.detach().cpu(),
        "attentions": attentions
    }

def generate_no_cache(prompt: str, gen_len: int = 20, max_prompt_len: int = 64) -> Dict:
    tokenizer = get_tokenizer()
    model = get_model()

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len)
    input_ids = enc["input_ids"].to(DEVICE).clone()

    generated = input_ids
    times = []
    for _ in range(gen_len):
        t0 = time.time()
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits  # (1, seq, vocab)
            next_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_id], dim=-1)
        t1 = time.time()
        times.append(t1 - t0)
    generated_ids = generated.detach().cpu()
    gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return {
        "generated_ids": generated_ids,
        "generated_text": gen_text,
        "times_per_step_s": times
    }

def generate_with_cache(prompt: str, gen_len: int = 20, max_prompt_len: int = 64) -> Dict:
    tokenizer = get_tokenizer()
    model = get_model()

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len)
    input_ids = enc["input_ids"].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past = outputs.past_key_values
        logits = outputs.logits
        next_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated = torch.cat([input_ids, next_id], dim=-1)
        times = [0.0]  # initial step is not timed meaningfully here

    for _ in range(1, gen_len):
        last = generated[:, -1:].to(DEVICE)
        t0 = time.time()
        with torch.no_grad():
            outputs = model(last, past_key_values=past, use_cache=True)
            logits = outputs.logits
            past = outputs.past_key_values
            next_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_id], dim=-1)
        t1 = time.time()
        times.append(t1 - t0)

    generated_ids = generated.detach().cpu()
    gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return {
        "generated_ids": generated_ids,
        "generated_text": gen_text,
        "times_per_step_s": times
    }

def get_stepwise_attn_snapshots(prompt: str, gen_len: int = 10, max_prompt_len=64) -> Dict:
    tokenizer = get_tokenizer()
    model = get_model()

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len)
    input_ids = enc["input_ids"].to(DEVICE)

    # NO CACHE snapshots
    no_cache_attns = []
    gen_ids_nc = input_ids.clone()
    for _ in range(gen_len):
        with torch.no_grad():
            out = model(gen_ids_nc, output_attentions=True)
            attn = out.attentions  # tuple len L; each (1, heads, seq, seq)
            # stack and take last-token rows
            # Convert to tensor: (L, heads, seq)
            step = torch.stack([layer[0, :, -1, :].cpu() for layer in attn], dim=0)
            no_cache_attns.append(step)
            logits = out.logits
            next_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            gen_ids_nc = torch.cat([gen_ids_nc, next_id], dim=-1)

    # CACHE snapshots
    cache_attns = []
    gen_ids_c = input_ids.clone()
    with torch.no_grad():
        out0 = model(gen_ids_c, output_attentions=True, use_cache=True)
        past = out0.past_key_values
        step0 = torch.stack([layer[0, :, -1, :].cpu() for layer in out0.attentions], dim=0)
        cache_attns.append(step0)
        logits0 = out0.logits
        next_id = torch.argmax(logits0[:, -1, :], dim=-1).unsqueeze(-1)
        gen_ids_c = torch.cat([gen_ids_c, next_id], dim=-1)

    for _ in range(1, gen_len):
        last = gen_ids_c[:, -1:].to(DEVICE)
        with torch.no_grad():
            out = model(last, past_key_values=past, use_cache=True, output_attentions=True)
            past = out.past_key_values
            logits = out.logits
            next_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            gen_ids_c = torch.cat([gen_ids_c, next_id], dim=-1)

        # capture attention of this full prefix
        with torch.no_grad():
            out_full = model(gen_ids_c, output_attentions=True)
            step = torch.stack([layer[0, :, -1, :].cpu() for layer in out_full.attentions], dim=0)
            cache_attns.append(step)

    all_ids = gen_ids_c.detach().cpu()[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(all_ids)
    return {
        "tokens": tokens,
        "attn_no_cache": no_cache_attns,     # list of L×heads×seq
        "attn_cache": cache_attns
    }
