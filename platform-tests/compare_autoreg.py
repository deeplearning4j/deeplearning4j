#!/usr/bin/env python3
"""Test autoregressive generation with raw ONNX models using onnxruntime.
Verifies whether the KV cache loop works correctly for step 1+."""

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
import os

MODEL_DIR = os.path.expanduser("~/.cache/dl4j-vlm-models")

def main():
    print("=== ONNX Autoregressive Generation Test ===")

    # Load tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "smoldocling-tokenizer.json"))
    image_token_id = tokenizer.token_to_id("<image>")

    # Load models
    decoder_sess = ort.InferenceSession(
        os.path.join(MODEL_DIR, "smoldocling-decoder.onnx"),
        providers=['CPUExecutionProvider']
    )
    embed_sess = ort.InferenceSession(
        os.path.join(MODEL_DIR, "smoldocling-embed-tokens.onnx"),
        providers=['CPUExecutionProvider']
    )

    # Use the saved merged embeddings from the previous Python test
    merged = np.load("/tmp/python_merged_embeddings.npy")
    print(f"Loaded merged embeddings: shape={merged.shape}")

    seq_len = merged.shape[1]
    num_layers = 30
    num_kv_heads = 3
    head_dim = 64

    # === Step 0: Prefill ===
    print(f"\n=== Step 0: Prefill (seq_len={seq_len}) ===")

    decoder_inputs = {
        'inputs_embeds': merged.astype(np.float32),
        'attention_mask': np.ones((1, seq_len), dtype=np.int64),
        'position_ids': np.arange(seq_len, dtype=np.int64).reshape(1, -1),
    }
    for i in range(num_layers):
        decoder_inputs[f'past_key_values.{i}.key'] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float32)
        decoder_inputs[f'past_key_values.{i}.value'] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float32)

    # Get output names
    output_names = [o.name for o in decoder_sess.get_outputs()]
    decoder_outputs = decoder_sess.run(output_names, decoder_inputs)

    # Parse outputs
    output_dict = dict(zip(output_names, decoder_outputs))
    logits = output_dict['logits']
    print(f"Logits: shape={logits.shape}")

    last_logits = logits[0, -1, :]
    next_token = np.argmax(last_logits)
    next_text = tokenizer.decode([int(next_token)])
    print(f"Step 0: token_id={next_token}, logit={last_logits[next_token]:.4f}, text='{next_text}'")
    print(f"  <doctag> (49229): logit={last_logits[49229]:.4f}")

    # Store KV cache
    kv_cache = {}
    for name in output_names:
        if name.startswith('present.'):
            kv_cache[name] = output_dict[name]
            if '0.key' in name:
                print(f"  KV cache '{name}': shape={output_dict[name].shape}")

    # === Steps 1-19: Autoregressive ===
    generated_tokens = [int(next_token)]
    past_seq_len = seq_len

    for step in range(1, 20):
        token_id = generated_tokens[-1]

        # Get embedding for new token
        embed_out = embed_sess.run(None, {'input_ids': np.array([[token_id]], dtype=np.int64)})
        new_embedding = embed_out[0]  # [1, 1, 576]

        # Build decoder inputs
        total_seq_len = past_seq_len + 1
        decoder_inputs = {
            'inputs_embeds': new_embedding.astype(np.float32),
            'attention_mask': np.ones((1, total_seq_len), dtype=np.int64),
            'position_ids': np.array([[past_seq_len]], dtype=np.int64),
        }

        # Use KV cache from previous step
        for i in range(num_layers):
            key_name = f'present.{i}.key'
            val_name = f'present.{i}.value'
            decoder_inputs[f'past_key_values.{i}.key'] = kv_cache[key_name]
            decoder_inputs[f'past_key_values.{i}.value'] = kv_cache[val_name]

        # Run decoder
        decoder_outputs = decoder_sess.run(output_names, decoder_inputs)
        output_dict = dict(zip(output_names, decoder_outputs))

        logits = output_dict['logits']
        last_logits = logits[0, -1, :]
        next_token = np.argmax(last_logits)
        next_text = tokenizer.decode([int(next_token)])

        print(f"Step {step}: token_id={next_token}, logit={last_logits[next_token]:.4f}, text='{next_text}'")

        generated_tokens.append(int(next_token))

        # Update KV cache
        for name in output_names:
            if name.startswith('present.'):
                kv_cache[name] = output_dict[name]

        past_seq_len = total_seq_len

        # Check for EOS
        if next_token == 2 or next_token == 49279:
            print("EOS generated")
            break

    # Decode all
    full_text = tokenizer.decode(generated_tokens)
    print(f"\nFull generated text: '{full_text}'")
    print(f"Token IDs: {generated_tokens}")

    print("\n=== ONNX Autoregressive Test Complete ===")

if __name__ == "__main__":
    main()
