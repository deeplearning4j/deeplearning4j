#!/usr/bin/env python3
"""Extract intermediate values from decoder for comparison with SameDiff."""
import os, sys, numpy as np

try:
    import onnxruntime as ort
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("pip install onnxruntime numpy onnx")
    sys.exit(1)

CACHE = os.path.expanduser("~/.cache/dl4j-vlm-models")
DECODER = os.path.join(CACHE, "smoldocling-decoder.onnx")
EMBED = os.path.join(CACHE, "smoldocling-embed-tokens.onnx")
TOK_JSON = os.path.join(CACHE, "smoldocling-tokenizer.json")

def main():
    print("Loading ONNX model...")
    model = onnx.load(DECODER)
    graph = model.graph

    # Build a map from output tensor name to its type
    output_type_map = {}
    for node in graph.node:
        for output in node.output:
            # Try to infer type from value_info
            for vi in graph.value_info:
                if vi.name == output:
                    output_type_map[output] = vi.type.tensor_type.elem_type
                    break

    # Also check initializers and graph inputs
    for inp in graph.input:
        output_type_map[inp.name] = inp.type.tensor_type.elem_type

    # Targets to extract - with type hints
    targets = {
        '/model/attn_mask_reformat/attn_mask_subgraph/Where_2/output_0': TensorProto.FLOAT,
        '/model/attn_mask_reformat/input_ids_subgraph/Where_2/output_0': TensorProto.FLOAT,
        '/model/attn_mask_reformat/input_ids_subgraph/Less/output_0': TensorProto.BOOL,
        '/model/attn_mask_reformat/input_ids_subgraph/Range/output_0': TensorProto.INT64,
        '/model/attn_mask_reformat/input_ids_subgraph/Expand/output_0': TensorProto.FLOAT,
        '/model/attn_mask_reformat/attn_mask_subgraph/Expand/output_0': TensorProto.INT64,
        '/model/attn_mask_reformat/input_ids_subgraph/output_0/output_0': TensorProto.INT64,
        '/model/layers.0/input_layernorm/output_0': TensorProto.FLOAT,
        '/model/layers.0/attn/q_proj/MatMul/output_0': TensorProto.FLOAT,
        '/model/layers.0/attn/k_proj/MatMul/output_0': TensorProto.FLOAT,
        '/model/layers.0/attn/o_proj/MatMul/output_0': TensorProto.FLOAT,
    }

    # Verify which exist
    all_outputs = set()
    for node in graph.node:
        for output in node.output:
            all_outputs.add(output)

    found = {k: v for k, v in targets.items() if k in all_outputs}
    missing = set(targets.keys()) - set(found.keys())
    for m in missing:
        print(f"  WARNING: '{m}' not found")
    print(f"Found {len(found)} intermediate outputs")

    # Add to graph
    for name, dtype in found.items():
        graph.output.append(helper.make_tensor_value_info(name, dtype, None))

    modified_path = os.path.join(CACHE, "decoder-intermediates.onnx")
    onnx.save(model, modified_path)

    # Load sessions
    print("Loading embed_tokens...")
    embed_sess = ort.InferenceSession(EMBED, providers=['CPUExecutionProvider'])
    print("Loading modified decoder...")
    decoder_sess = ort.InferenceSession(modified_path, providers=['CPUExecutionProvider'])
    decoder_inputs = {i.name: i for i in decoder_sess.get_inputs()}

    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(TOK_JSON)
    except:
        tokenizer = None

    prompt = "<|im_start|>User:Convert this page to docling.<end_of_utterance>\nAssistant:"
    if tokenizer:
        token_ids = tokenizer.encode(prompt).ids
    else:
        token_ids = [1, 11126, 42, 37983, 451, 3013, 288, 2226, 1519, 30, 49279, 198, 9519, 9531, 42]

    print(f"\nToken IDs ({len(token_ids)}): {token_ids}")

    input_ids = np.array([token_ids], dtype=np.int64)
    embeddings = embed_sess.run(None, {embed_sess.get_inputs()[0].name: input_ids})[0]

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
                    shape.append(1)
                elif i == 2:
                    shape.append(0)
                else:
                    shape.append(0)
            feed[name] = np.zeros(shape, dtype=np.float32)

    output_names = ['logits'] + list(found.keys())
    print(f"Requesting {len(output_names)} outputs...")
    results = decoder_sess.run(output_names, feed)
    results_map = dict(zip(output_names, results))

    # Print logits
    logits = results_map['logits']
    last_logits = logits[0, -1, :]
    print(f"\n=== Logits ===")
    print(f"Shape: {logits.shape}")
    print(f"Stats: min={last_logits.min():.4f}, max={last_logits.max():.4f}, mean={last_logits.mean():.4f}")
    top_idx = np.argsort(last_logits)[::-1][:5]
    for i, idx in enumerate(top_idx):
        tok = tokenizer.decode([idx]) if tokenizer else f"token_{idx}"
        print(f"  #{i+1}: id={idx}, logit={last_logits[idx]:.4f}, text='{tok}'")

    # Print intermediates
    for name in found.keys():
        arr = results_map[name]
        print(f"\n=== {name} ===")
        print(f"  shape: {arr.shape}, dtype: {arr.dtype}")
        if arr.size == 0:
            print(f"  EMPTY TENSOR")
        elif arr.size <= 200:
            if arr.ndim <= 2:
                print(f"  values:\n{arr}")
            else:
                print(f"  values (flat[:50]): {arr.flat[:50]}")
        else:
            print(f"  min: {arr.min()}, max: {arr.max()}, mean: {arr.mean():.6f}")
            if arr.ndim >= 2:
                print(f"  first row[:10]: {arr.flat[:10]}")
                print(f"  last 10: {arr.flat[-10:]}")
            if arr.ndim == 4:
                print(f"  [0,0,0,:5]: {arr[0,0,0,:5]}")
                print(f"  [0,0,-1,:5]: {arr[0,0,-1,:5]}")

    # Cleanup
    os.remove(modified_path)

if __name__ == "__main__":
    main()
