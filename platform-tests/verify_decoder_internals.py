#!/usr/bin/env python3
"""
Compare SameDiff vs onnxruntime decoder internals.
Feeds same text input through embed_tokens + decoder and dumps intermediates.
"""

import os, sys, numpy as np

try:
    import onnxruntime as ort
    import onnx
except ImportError:
    print("pip install onnxruntime numpy onnx")
    sys.exit(1)

CACHE = os.path.expanduser("~/.cache/dl4j-vlm-models")
DECODER = os.path.join(CACHE, "smoldocling-decoder.onnx")
EMBED = os.path.join(CACHE, "smoldocling-embed-tokens.onnx")
TOK_JSON = os.path.join(CACHE, "smoldocling-tokenizer.json")

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def main():
    for p in [DECODER, EMBED]:
        if not os.path.exists(p):
            print(f"Missing: {p}")
            sys.exit(1)

    # Load tokenizer
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(TOK_JSON)
    except:
        tokenizer = None

    # === Load models ===
    print("Loading embed_tokens...")
    embed_sess = ort.InferenceSession(EMBED, providers=['CPUExecutionProvider'])

    print("Loading decoder...")
    decoder_sess = ort.InferenceSession(DECODER, providers=['CPUExecutionProvider'])
    decoder_inputs = {i.name: i for i in decoder_sess.get_inputs()}

    print(f"Decoder inputs: {list(decoder_inputs.keys())}")
    for name, inp in decoder_inputs.items():
        print(f"  {name}: shape={inp.shape}, dtype={inp.type}")

    # === Test: DocTag prompt (matches Java test) ===
    prompt = "<|im_start|>User:Convert this page to docling.<end_of_utterance>\nAssistant:"
    print(f"\nPrompt: {prompt}")

    if tokenizer:
        token_ids = tokenizer.encode(prompt).ids
    else:
        token_ids = [1, 11126, 42, 18099, 436, 3013, 288, 2226, 1519, 30, 49279, 198, 9519, 9531, 42]

    print(f"Token IDs ({len(token_ids)}): {token_ids}")

    # Get embeddings
    input_ids = np.array([token_ids], dtype=np.int64)
    embeddings = embed_sess.run(None, {embed_sess.get_inputs()[0].name: input_ids})[0]
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    print(f"Embeddings range: [{embeddings.min():.6f}, {embeddings.max():.6f}]")
    print(f"Embeddings mean: {embeddings.mean():.6f}")
    print(f"Embeddings[0,0,:5]: {embeddings[0,0,:5]}")
    print(f"Embeddings[0,-1,:5]: {embeddings[0,-1,:5]}")

    # RMS LayerNorm manually (to compare with SameDiff)
    emb = embeddings[0]  # [seq_len, 576]
    sq = emb ** 2
    rms_mean = sq.mean(axis=-1, keepdims=True)  # [seq_len, 1]
    rms_div = np.sqrt(rms_mean + 1e-5)  # epsilon = 1e-5 typically
    rms_normalized = emb / rms_div
    print(f"\n--- Manual RMS LayerNorm (eps=1e-5) ---")
    print(f"RMS divisor pos0: {rms_div[0,0]:.6f}")
    print(f"RMS divisor pos5: {rms_div[min(5, len(rms_div)-1),0]:.6f}")
    print(f"RMS divisor posLast: {rms_div[-1,0]:.6f}")
    print(f"Normalized[0,:5]: {rms_normalized[0,:5]}")
    print(f"Normalized[-1,:5]: {rms_normalized[-1,:5]}")

    # Build decoder inputs
    seq_len = embeddings.shape[1]
    feed = {}
    for name, inp in decoder_inputs.items():
        if name == "inputs_embeds":
            feed[name] = embeddings.astype(np.float32)
        elif name == "attention_mask":
            feed[name] = np.ones((1, seq_len), dtype=np.int64)
        elif name == "position_ids":
            feed[name] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        elif name.startswith("past_key_values"):
            shape = []
            for i, d in enumerate(inp.shape):
                if isinstance(d, int) and d > 0:
                    shape.append(d)
                elif i == 0:
                    shape.append(1)  # batch
                elif i == 2:
                    shape.append(0)  # past_seq_len = 0
                else:
                    shape.append(0)
            feed[name] = np.zeros(shape, dtype=np.float32)

    print(f"\n--- Running decoder (seq_len={seq_len}) ---")
    print(f"inputs_embeds shape: {feed['inputs_embeds'].shape}")
    print(f"attention_mask: {feed['attention_mask']}")
    print(f"position_ids: {feed['position_ids']}")

    # Get all output names
    output_names = [o.name for o in decoder_sess.get_outputs()]
    logits_name = next((n for n in output_names if 'logit' in n.lower()), output_names[0])

    # Run decoder - get logits only first
    results = decoder_sess.run([logits_name], feed)
    logits = results[0]

    print(f"\nLogits shape: {logits.shape}")
    last_logits = logits[0, -1, :] if logits.ndim == 3 else logits[0, :]
    print(f"Logits stats: min={last_logits.min():.6f}, max={last_logits.max():.6f}, mean={last_logits.mean():.6f}")

    probs = softmax(last_logits)
    top_idx = np.argsort(last_logits)[::-1][:10]
    print("\nTop-10 predictions (text-only, no image):")
    for i, idx in enumerate(top_idx):
        tok_text = tokenizer.decode([idx]) if tokenizer else f"token_{idx}"
        print(f"  #{i+1}: id={idx}, logit={last_logits[idx]:.6f}, prob={probs[idx]:.8f}, text='{tok_text}'")

    print(f"\n<doctag> (49229): logit={last_logits[49229]:.6f}, prob={probs[49229]:.2e}")
    print(f"<end_of_utterance> (49279): logit={last_logits[49279]:.6f}, prob={probs[49279]:.2e}")
    print(f"\\n (198): logit={last_logits[198]:.6f}, prob={probs[198]:.2e}")

    # === Now compare with Java's vision prompt ===
    # Java's prompt has 283 tokens = text tokens + 4 tiles * 64 image tokens = 27 text + 256 image
    # Let's also test the "Hello" prompt from earlier
    print("\n\n=== Test 2: Hello prompt ===")
    prompt2 = "<|im_start|>User:Hello, how are you?<end_of_utterance>\nAssistant:"
    if tokenizer:
        token_ids2 = tokenizer.encode(prompt2).ids
    else:
        token_ids2 = [1, 11126, 42, 19556, 28, 638, 359, 346, 47, 49279, 198, 9519, 9531, 42]

    print(f"Token IDs ({len(token_ids2)}): {token_ids2}")
    input_ids2 = np.array([token_ids2], dtype=np.int64)
    embeddings2 = embed_sess.run(None, {embed_sess.get_inputs()[0].name: input_ids2})[0]

    seq_len2 = embeddings2.shape[1]
    feed2 = {}
    for name, inp in decoder_inputs.items():
        if name == "inputs_embeds":
            feed2[name] = embeddings2.astype(np.float32)
        elif name == "attention_mask":
            feed2[name] = np.ones((1, seq_len2), dtype=np.int64)
        elif name == "position_ids":
            feed2[name] = np.arange(seq_len2, dtype=np.int64).reshape(1, -1)
        elif name.startswith("past_key_values"):
            shape = []
            for i, d in enumerate(inp.shape):
                if isinstance(d, int) and d > 0:
                    shape.append(d)
                elif i == 0:
                    shape.append(1)
                elif i == 2:
                    shape.append(0)
                else:
                    shape.append(0)
            feed2[name] = np.zeros(shape, dtype=np.float32)

    logits2 = decoder_sess.run([logits_name], feed2)[0]
    last_logits2 = logits2[0, -1, :]
    print(f"Logits stats: min={last_logits2.min():.6f}, max={last_logits2.max():.6f}, mean={last_logits2.mean():.6f}")

    probs2 = softmax(last_logits2)
    top_idx2 = np.argsort(last_logits2)[::-1][:10]
    print("Top-10:")
    for i, idx in enumerate(top_idx2):
        tok_text = tokenizer.decode([idx]) if tokenizer else f"token_{idx}"
        print(f"  #{i+1}: id={idx}, logit={last_logits2[idx]:.6f}, prob={probs2[idx]:.8f}, text='{tok_text}'")

    # === Dump first few embedding values for exact comparison ===
    print("\n\n=== Exact embedding values for comparison ===")
    print(f"embed[0,0,:10] = {embeddings2[0,0,:10].tolist()}")
    print(f"embed[0,1,:10] = {embeddings2[0,1,:10].tolist()}")
    print(f"embed[0,-1,:10] = {embeddings2[0,-1,:10].tolist()}")

    # Manual RMS norm for hello prompt
    emb2 = embeddings2[0]
    sq2 = emb2 ** 2
    rms_mean2 = sq2.mean(axis=-1, keepdims=True)
    rms_div2 = np.sqrt(rms_mean2 + 1e-5)
    print(f"\nRMS divisor[0] = {rms_div2[0,0]:.10f}")
    print(f"RMS divisor[-1] = {rms_div2[-1,0]:.10f}")

    rms_norm2 = emb2 / rms_div2
    print(f"RMS normed[0,:10] = {rms_norm2[0,:10].tolist()}")
    print(f"RMS normed[-1,:10] = {rms_norm2[-1,:10].tolist()}")

if __name__ == "__main__":
    main()
