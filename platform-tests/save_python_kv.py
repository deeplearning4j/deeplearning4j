#!/usr/bin/env python3
"""Save Python ONNX KV cache and step 1 inputs for comparison with Java."""

import numpy as np
import onnxruntime as ort
from PIL import Image
import fitz
import os
from tokenizers import Tokenizer

MODEL_DIR = os.path.expanduser("~/.cache/dl4j-vlm-models")
PDF_PATH = "pathfinder-mythic.pdf"
PAGE_NUM = 10
TARGET_SIZE = 512

def load_tokenizer():
    return Tokenizer.from_file(os.path.join(MODEL_DIR, "smoldocling-tokenizer.json"))

def render_pdf_page(pdf_path, page_num, dpi=150):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def preprocess_frame(frame, target_size):
    arr = np.array(frame).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr.transpose(2, 0, 1)
    return arr.reshape(1, 1, 3, target_size, target_size)

def split_image(img, target_size, max_tiles):
    w, h = img.size
    cols = w // target_size
    rows = h // target_size
    while rows * cols > max_tiles:
        if rows >= cols: rows -= 1
        else: cols -= 1
    tile_w, tile_h = w // cols, h // rows
    frames = []
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * tile_w, r * tile_h
            x1, y1 = min(x0 + tile_w, w), min(y0 + tile_h, h)
            tile = img.crop((x0, y0, x1, y1)).resize((target_size, target_size), Image.LANCZOS)
            frames.append(tile)
    frames.append(img.resize((target_size, target_size), Image.LANCZOS))
    return frames, rows, cols

def build_image_prompt(num_rows, num_cols, seq_len):
    fake = "<fake_token_around_image>"
    image = "<image>"
    glob = "<global-img>"
    sb = []
    for r in range(1, num_rows + 1):
        for c in range(1, num_cols + 1):
            sb.append(fake + f"<row_{r}_col_{c}>" + image * seq_len)
        sb.append("\n")
    sb.append("\n" + fake + glob + image * seq_len + fake)
    return "".join(sb)

def resize_for_tiling(img, target_size):
    w, h = img.size
    aspect = w / h
    new_w = max(target_size, ((w + target_size - 1) // target_size) * target_size)
    new_h = max(target_size, ((h + target_size - 1) // target_size) * target_size)
    if aspect >= 1:
        new_h = max(target_size, round(new_w / aspect / target_size) * target_size)
    else:
        new_w = max(target_size, round(new_h * aspect / target_size) * target_size)
    return img.resize((new_w, new_h), Image.LANCZOS)

def main():
    print("=== Save Python KV Cache Reference ===")
    tokenizer = load_tokenizer()
    image_token_id = tokenizer.token_to_id("<image>")

    vision_sess = ort.InferenceSession(os.path.join(MODEL_DIR, "smoldocling-vision-encoder.onnx"))
    decoder_sess = ort.InferenceSession(os.path.join(MODEL_DIR, "smoldocling-decoder.onnx"))
    embed_sess = ort.InferenceSession(os.path.join(MODEL_DIR, "smoldocling-embed-tokens.onnx"))

    img = render_pdf_page(PDF_PATH, PAGE_NUM)
    img_resized = resize_for_tiling(img, TARGET_SIZE)
    frames, num_rows, num_cols = split_image(img_resized, TARGET_SIZE, 4)

    frame_embeddings = []
    for frame in frames:
        pv = preprocess_frame(frame, TARGET_SIZE)
        mask = np.ones((1, 1, TARGET_SIZE, TARGET_SIZE), dtype=np.bool_)
        outputs = vision_sess.run(None, {'pixel_values': pv, 'pixel_attention_mask': mask})
        frame_embeddings.append(outputs[0])
    vision_embeddings = np.concatenate(frame_embeddings, axis=1)

    image_seq_len = frame_embeddings[0].shape[1]
    image_prompt = build_image_prompt(num_rows, num_cols, image_seq_len)
    chat_prompt = f"<|im_start|>User:{image_prompt}Convert this page to docling.<end_of_utterance>\nAssistant:"
    prompt_token_ids = tokenizer.encode(chat_prompt).ids
    seq_len = len(prompt_token_ids)

    input_ids = np.array([prompt_token_ids], dtype=np.int64)
    text_embeddings = embed_sess.run(None, {'input_ids': input_ids})[0]

    merged = text_embeddings.copy()
    fill_idx = 0
    vision_flat = vision_embeddings.reshape(-1, vision_embeddings.shape[2])
    for pos in range(seq_len):
        if prompt_token_ids[pos] == image_token_id and fill_idx < vision_flat.shape[0]:
            merged[0, pos, :] = vision_flat[fill_idx]
            fill_idx += 1

    # Step 0: Prefill
    num_layers, num_kv_heads, head_dim = 30, 3, 64
    decoder_inputs = {
        'inputs_embeds': merged.astype(np.float32),
        'attention_mask': np.ones((1, seq_len), dtype=np.int64),
        'position_ids': np.arange(seq_len, dtype=np.int64).reshape(1, -1),
    }
    for i in range(num_layers):
        decoder_inputs[f'past_key_values.{i}.key'] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float32)
        decoder_inputs[f'past_key_values.{i}.value'] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float32)

    output_names = [o.name for o in decoder_sess.get_outputs()]
    outputs = decoder_sess.run(output_names, decoder_inputs)
    output_map = dict(zip(output_names, outputs))

    logits = output_map['logits']
    step0_token = int(np.argmax(logits[0, -1, :]))
    print(f"Step 0: token={step0_token}")

    # Save key diagnostic values
    # KV cache layer 0 key - first few values
    kv0_key = output_map['present.0.key']
    kv0_val = output_map['present.0.value']
    print(f"present.0.key: shape={kv0_key.shape}, min={kv0_key.min():.6f}, max={kv0_key.max():.6f}")
    print(f"present.0.key[0,0,0,:5] = {kv0_key[0,0,0,:5]}")
    print(f"present.0.key[0,0,-1,:5] = {kv0_key[0,0,-1,:5]}")
    print(f"present.0.key[0,1,0,:5] = {kv0_key[0,1,0,:5]}")
    print(f"present.0.value[0,0,0,:5] = {kv0_val[0,0,0,:5]}")

    # Also check layer 15 (middle)
    kv15_key = output_map['present.15.key']
    print(f"present.15.key[0,0,0,:5] = {kv15_key[0,0,0,:5]}")

    # Step 1 embedding
    new_token_embed = embed_sess.run(None, {'input_ids': np.array([[step0_token]], dtype=np.int64)})[0]
    print(f"\nStep 1 embedding (token {step0_token}): shape={new_token_embed.shape}")
    print(f"Step 1 embedding[0,0,:10] = {new_token_embed[0,0,:10]}")

    # Step 1 logits
    step1_inputs = {
        'inputs_embeds': new_token_embed.astype(np.float32),
        'attention_mask': np.ones((1, seq_len + 1), dtype=np.int64),
        'position_ids': np.array([[seq_len]], dtype=np.int64),
    }
    for i in range(num_layers):
        step1_inputs[f'past_key_values.{i}.key'] = output_map[f'present.{i}.key']
        step1_inputs[f'past_key_values.{i}.value'] = output_map[f'present.{i}.value']

    step1_outputs = decoder_sess.run(output_names, step1_inputs)
    step1_map = dict(zip(output_names, step1_outputs))
    step1_logits = step1_map['logits']
    step1_last = step1_logits[0, -1, :]
    step1_token = int(np.argmax(step1_last))
    print(f"\nStep 1: token={step1_token}, text='{tokenizer.decode([step1_token])}'")
    print(f"Step 1 logits[0,-1,0:10] = {step1_last[:10]}")
    print(f"Step 1 <doctag> (49229): logit={step1_last[49229]:.4f}")
    print(f"Step 1 tab (197): logit={step1_last[197]:.4f}")

    # Save reference values
    np.save("/tmp/python_kv0_key.npy", kv0_key)
    np.save("/tmp/python_step1_logits.npy", step1_logits)
    np.save("/tmp/python_step1_embed.npy", new_token_embed)
    print("\nSaved reference files to /tmp/")
    print("=== Done ===")

if __name__ == "__main__":
    main()
