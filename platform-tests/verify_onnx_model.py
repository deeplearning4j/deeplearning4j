#!/usr/bin/env python3
"""
Verify SmolDocling ONNX model outputs using onnxruntime.
Compares Python/onnxruntime reference with SameDiff implementation.

Usage:
  pip install onnxruntime numpy transformers pillow
  python verify_onnx_model.py

This loads the cached ONNX models and runs a simple text-only test,
then a vision test, printing the top-10 logits from the decoder.
"""

import os
import sys
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("Please install: pip install onnxruntime numpy")
    sys.exit(1)

CACHE_DIR = os.path.expanduser("~/.cache/dl4j-vlm-models")
VISION_ENCODER = os.path.join(CACHE_DIR, "smoldocling-vision-encoder.onnx")
DECODER = os.path.join(CACHE_DIR, "smoldocling-decoder.onnx")
EMBED_TOKENS = os.path.join(CACHE_DIR, "smoldocling-embed-tokens.onnx")
TOKENIZER_JSON = os.path.join(CACHE_DIR, "smoldocling-tokenizer.json")

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def build_causal_mask(seq_len, dtype=np.float32):
    """Build lower-triangular causal mask [1, 1, Q, K]"""
    MASK_FILL = np.float32(-3.4028235e+38)
    mask = np.zeros((1, 1, seq_len, seq_len), dtype=dtype)
    for q in range(seq_len):
        for k in range(q + 1, seq_len):
            mask[0, 0, q, k] = MASK_FILL
    return mask

def main():
    # Check model files
    for path, name in [(DECODER, "decoder"), (EMBED_TOKENS, "embed_tokens")]:
        if not os.path.exists(path):
            print(f"Model not found: {path}")
            print(f"Run the Java test first to download models, or download manually.")
            sys.exit(1)

    print("Loading embed_tokens model...")
    embed_sess = ort.InferenceSession(EMBED_TOKENS, providers=['CPUExecutionProvider'])
    print(f"  Inputs: {[i.name for i in embed_sess.get_inputs()]}")
    print(f"  Outputs: {[o.name for o in embed_sess.get_outputs()]}")

    print("Loading decoder model...")
    decoder_sess = ort.InferenceSession(DECODER, providers=['CPUExecutionProvider'])
    decoder_inputs = {i.name: i for i in decoder_sess.get_inputs()}
    decoder_outputs = {o.name: o for o in decoder_sess.get_outputs()}
    print(f"  Inputs: {list(decoder_inputs.keys())}")
    print(f"  Outputs: {list(decoder_outputs.keys())}")

    # Try to load tokenizer
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(TOKENIZER_JSON)
        print(f"Tokenizer loaded: vocab_size={tokenizer.get_vocab_size()}")
    except ImportError:
        print("Install 'tokenizers' package for tokenizer support: pip install tokenizers")
        tokenizer = None

    # === Test 1: Text-only "Hello" ===
    print("\n=== Test 1: Text-only 'Hello' ===")
    prompt = "<|im_start|>User:Hello, how are you?<end_of_utterance>\nAssistant:"

    if tokenizer:
        encoding = tokenizer.encode(prompt)
        token_ids = encoding.ids
    else:
        # Fallback: manually specified token IDs from Java test output
        token_ids = [1, 11126, 42, 19556, 28, 638, 359, 346, 47, 49279, 198, 9519, 9531, 42]

    print(f"Token IDs ({len(token_ids)}): {token_ids[:20]}...")

    # Get embeddings
    input_ids = np.array([token_ids], dtype=np.int64)
    embed_input_name = embed_sess.get_inputs()[0].name
    embed_output_name = embed_sess.get_outputs()[0].name
    embeddings = embed_sess.run([embed_output_name], {embed_input_name: input_ids})[0]
    print(f"Embeddings shape: {embeddings.shape}, range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")

    # Build decoder inputs
    seq_len = embeddings.shape[1]
    decoder_feed = {}
    for name, inp in decoder_inputs.items():
        if name == "inputs_embeds":
            decoder_feed[name] = embeddings.astype(np.float32)
        elif name == "attention_mask":
            decoder_feed[name] = np.ones((1, seq_len), dtype=np.int64)
        elif name == "position_ids":
            decoder_feed[name] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        elif name.startswith("past_key_values") and name.endswith(".key"):
            # Shape from input spec: [batch, num_kv_heads, 0, head_dim]
            shape = [d if isinstance(d, int) and d > 0 else (1 if i == 0 else 0) for i, d in enumerate(inp.shape)]
            shape[2] = 0  # seq_len = 0 for first step
            decoder_feed[name] = np.zeros(shape, dtype=np.float32)
        elif name.startswith("past_key_values") and name.endswith(".value"):
            shape = [d if isinstance(d, int) and d > 0 else (1 if i == 0 else 0) for i, d in enumerate(inp.shape)]
            shape[2] = 0
            decoder_feed[name] = np.zeros(shape, dtype=np.float32)
        elif name == "input_ids":
            decoder_feed[name] = input_ids
        else:
            print(f"  WARNING: Unknown input '{name}', shape={inp.shape}, dtype={inp.type}")

    # Run decoder - only get logits
    logits_name = None
    for o in decoder_sess.get_outputs():
        if "logit" in o.name.lower():
            logits_name = o.name
            break
    if logits_name is None:
        logits_name = decoder_sess.get_outputs()[0].name

    print(f"Running decoder for logits output: {logits_name}")
    logits = decoder_sess.run([logits_name], decoder_feed)[0]
    print(f"Logits shape: {logits.shape}")

    # Get last position logits
    if logits.ndim == 3:
        last_logits = logits[0, -1, :]
    else:
        last_logits = logits[0, :]

    print(f"Last logits: min={last_logits.min():.4f}, max={last_logits.max():.4f}, mean={last_logits.mean():.4f}")

    # Top-10
    probs = softmax(last_logits)
    top_indices = np.argsort(last_logits)[::-1][:10]
    print("Top-10 predictions:")
    for i, idx in enumerate(top_indices):
        tok_text = tokenizer.decode([idx]) if tokenizer else f"token_{idx}"
        print(f"  #{i+1}: id={idx}, logit={last_logits[idx]:.4f}, prob={probs[idx]:.6f}, text='{tok_text}'")

    # Check specific tokens
    print(f"\n<doctag> (49229): logit={last_logits[49229]:.4f}, prob={probs[49229]:.2e}")
    print(f"<end_of_utterance> (49279): logit={last_logits[49279]:.4f}, prob={probs[49279]:.2e}")

    # === Test 2: DocTag prompt ===
    print("\n=== Test 2: DocTag prompt (no image) ===")
    prompt2 = "<|im_start|>User:Convert this page to docling.<end_of_utterance>\nAssistant:"
    if tokenizer:
        token_ids2 = tokenizer.encode(prompt2).ids
    else:
        token_ids2 = [1, 11126, 42, 18099, 436, 3013, 288, 2226, 1519, 30, 49279, 198, 9519, 9531, 42]

    input_ids2 = np.array([token_ids2], dtype=np.int64)
    embeddings2 = embed_sess.run([embed_output_name], {embed_input_name: input_ids2})[0]
    print(f"Embeddings shape: {embeddings2.shape}")

    seq_len2 = embeddings2.shape[1]
    decoder_feed2 = {}
    for name, inp in decoder_inputs.items():
        if name == "inputs_embeds":
            decoder_feed2[name] = embeddings2.astype(np.float32)
        elif name == "attention_mask":
            decoder_feed2[name] = np.ones((1, seq_len2), dtype=np.int64)
        elif name == "position_ids":
            decoder_feed2[name] = np.arange(seq_len2, dtype=np.int64).reshape(1, -1)
        elif name.startswith("past_key_values") and name.endswith(".key"):
            shape = [d if isinstance(d, int) and d > 0 else (1 if i == 0 else 0) for i, d in enumerate(inp.shape)]
            shape[2] = 0
            decoder_feed2[name] = np.zeros(shape, dtype=np.float32)
        elif name.startswith("past_key_values") and name.endswith(".value"):
            shape = [d if isinstance(d, int) and d > 0 else (1 if i == 0 else 0) for i, d in enumerate(inp.shape)]
            shape[2] = 0
            decoder_feed2[name] = np.zeros(shape, dtype=np.float32)
        elif name == "input_ids":
            decoder_feed2[name] = input_ids2

    logits2 = decoder_sess.run([logits_name], decoder_feed2)[0]
    last_logits2 = logits2[0, -1, :] if logits2.ndim == 3 else logits2[0, :]
    print(f"Logits: min={last_logits2.min():.4f}, max={last_logits2.max():.4f}, mean={last_logits2.mean():.4f}")

    probs2 = softmax(last_logits2)
    top_indices2 = np.argsort(last_logits2)[::-1][:10]
    print("Top-10:")
    for i, idx in enumerate(top_indices2):
        tok_text = tokenizer.decode([idx]) if tokenizer else f"token_{idx}"
        print(f"  #{i+1}: id={idx}, logit={last_logits2[idx]:.4f}, prob={probs2[idx]:.6f}, text='{tok_text}'")

    print(f"\n<doctag> (49229): logit={last_logits2[49229]:.4f}, prob={probs2[49229]:.2e}")
    print(f"<end_of_utterance> (49279): logit={last_logits2[49279]:.4f}, prob={probs2[49279]:.2e}")

    print("\n=== COMPARISON COMPLETE ===")
    print("Compare the above with the SameDiff test output in the surefire reports.")
    print("If Python produces different top tokens, the issue is in SameDiff import.")
    print("If Python produces the same wrong tokens, the issue is in the ONNX model.")

if __name__ == "__main__":
    main()
