#!/usr/bin/env python3
"""Test step 1 (incremental) with ONNX runtime to verify KV cache handling."""

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

    tile_w = w // cols
    tile_h = h // rows
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
            sb.append(fake)
            sb.append(f"<row_{r}_col_{c}>")
            sb.append(image * seq_len)
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
    print("=== Python ONNX 2-Step Reference ===")

    tokenizer = load_tokenizer()
    image_token_id = tokenizer.token_to_id("<image>")

    # Load ONNX models
    vision_sess = ort.InferenceSession(os.path.join(MODEL_DIR, "smoldocling-vision-encoder.onnx"))
    decoder_sess = ort.InferenceSession(os.path.join(MODEL_DIR, "smoldocling-decoder.onnx"))
    embed_sess = ort.InferenceSession(os.path.join(MODEL_DIR, "smoldocling-embed-tokens.onnx"))

    # Process image
    img = render_pdf_page(PDF_PATH, PAGE_NUM)
    img_resized = resize_for_tiling(img, TARGET_SIZE)
    frames, num_rows, num_cols = split_image(img_resized, TARGET_SIZE, 4)
    print(f"Frames: {len(frames)}")

    # Vision encoder
    frame_embeddings = []
    for i, frame in enumerate(frames):
        pv = preprocess_frame(frame, TARGET_SIZE)
        mask = np.ones((1, 1, TARGET_SIZE, TARGET_SIZE), dtype=np.bool_)
        outputs = vision_sess.run(None, {'pixel_values': pv, 'pixel_attention_mask': mask})
        frame_embeddings.append(outputs[0])
    vision_embeddings = np.concatenate(frame_embeddings, axis=1)
    print(f"Vision embeddings: {vision_embeddings.shape}")

    # Build prompt and tokenize
    image_seq_len = frame_embeddings[0].shape[1]
    image_prompt = build_image_prompt(num_rows, num_cols, image_seq_len)
    prompt_text = "Convert this page to docling."
    chat_prompt = f"<|im_start|>User:{image_prompt}{prompt_text}<end_of_utterance>\nAssistant:"
    encoding = tokenizer.encode(chat_prompt)
    prompt_token_ids = encoding.ids
    print(f"Tokens: {len(prompt_token_ids)}, <image> count: {sum(1 for t in prompt_token_ids if t == image_token_id)}")

    # Get text embeddings
    input_ids = np.array([prompt_token_ids], dtype=np.int64)
    text_embeddings = embed_sess.run(None, {'input_ids': input_ids})[0]

    # Merge
    seq_len = len(prompt_token_ids)
    hidden_dim = vision_embeddings.shape[2]
    merged = text_embeddings.copy()
    fill_idx = 0
    vision_flat = vision_embeddings.reshape(-1, hidden_dim)
    for pos in range(seq_len):
        if prompt_token_ids[pos] == image_token_id and fill_idx < vision_flat.shape[0]:
            merged[0, pos, :] = vision_flat[fill_idx]
            fill_idx += 1
    print(f"Merged: {merged.shape}, filled {fill_idx} image positions")

    # ===== STEP 0: Prefill =====
    print("\n=== STEP 0: Prefill ===")
    num_layers = 30
    num_kv_heads = 3
    head_dim = 64

    decoder_inputs = {
        'inputs_embeds': merged.astype(np.float32),
        'attention_mask': np.ones((1, seq_len), dtype=np.int64),
        'position_ids': np.arange(seq_len, dtype=np.int64).reshape(1, -1),
    }
    for i in range(num_layers):
        decoder_inputs[f'past_key_values.{i}.key'] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float32)
        decoder_inputs[f'past_key_values.{i}.value'] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float32)

    decoder_output_names = [o.name for o in decoder_sess.get_outputs()]
    decoder_outputs = decoder_sess.run(decoder_output_names, decoder_inputs)

    # Parse outputs
    output_map = dict(zip(decoder_output_names, decoder_outputs))
    logits = output_map['logits']
    last_logits = logits[0, -1, :]
    step0_token = np.argmax(last_logits)
    step0_text = tokenizer.decode([int(step0_token)])
    print(f"Step 0: token={step0_token}, text='{step0_text}', logit={last_logits[step0_token]:.4f}")
    print(f"  <doctag> (49229): logit={last_logits[49229]:.4f}")

    # Extract KV cache for step 1
    kv_cache = {}
    for name in decoder_output_names:
        if name.startswith("present."):
            kv_cache[name] = output_map[name]
            if '0.key' in name:
                print(f"  KV cache '{name}': shape={output_map[name].shape}")

    # ===== STEP 1: Incremental with KV cache =====
    print("\n=== STEP 1: Incremental ===")

    # Get embedding for the sampled token
    new_token_ids = np.array([[int(step0_token)]], dtype=np.int64)
    new_token_embed = embed_sess.run(None, {'input_ids': new_token_ids})[0]
    print(f"New token embedding: shape={new_token_embed.shape}")

    past_seq_len = seq_len  # 348
    new_seq_len = 1
    total_seq_len = past_seq_len + new_seq_len  # 349

    step1_inputs = {
        'inputs_embeds': new_token_embed.astype(np.float32),
        'attention_mask': np.ones((1, total_seq_len), dtype=np.int64),
        'position_ids': np.array([[past_seq_len]], dtype=np.int64),  # [348]
    }
    for i in range(num_layers):
        step1_inputs[f'past_key_values.{i}.key'] = kv_cache[f'present.{i}.key']
        step1_inputs[f'past_key_values.{i}.value'] = kv_cache[f'present.{i}.value']

    step1_outputs = decoder_sess.run(decoder_output_names, step1_inputs)
    step1_map = dict(zip(decoder_output_names, step1_outputs))

    step1_logits = step1_map['logits']
    print(f"Step 1 logits shape: {step1_logits.shape}")
    step1_last = step1_logits[0, -1, :]
    step1_token = np.argmax(step1_last)
    step1_text = tokenizer.decode([int(step1_token)])
    print(f"Step 1: token={step1_token}, text='{step1_text}', logit={step1_last[step1_token]:.4f}")
    print(f"  <doctag> (49229): logit={step1_last[49229]:.4f}")

    # Top-5 at step 1
    top5 = np.argsort(step1_last)[-5:][::-1]
    print("Step 1 top-5:")
    for tid in top5:
        print(f"  id={tid}, logit={step1_last[tid]:.4f}, text='{tokenizer.decode([int(tid)])}'")

    # ===== STEP 2: Another increment =====
    print("\n=== STEP 2 ===")
    step1_kv = {}
    for name in decoder_output_names:
        if name.startswith("present."):
            step1_kv[name] = step1_map[name]

    new2_token_ids = np.array([[int(step1_token)]], dtype=np.int64)
    new2_token_embed = embed_sess.run(None, {'input_ids': new2_token_ids})[0]

    step2_inputs = {
        'inputs_embeds': new2_token_embed.astype(np.float32),
        'attention_mask': np.ones((1, total_seq_len + 1), dtype=np.int64),
        'position_ids': np.array([[past_seq_len + 1]], dtype=np.int64),
    }
    for i in range(num_layers):
        step2_inputs[f'past_key_values.{i}.key'] = step1_kv[f'present.{i}.key']
        step2_inputs[f'past_key_values.{i}.value'] = step1_kv[f'present.{i}.value']

    step2_outputs = decoder_sess.run(decoder_output_names, step2_inputs)
    step2_map = dict(zip(decoder_output_names, step2_outputs))
    step2_logits = step2_map['logits']
    step2_last = step2_logits[0, -1, :]
    step2_token = np.argmax(step2_last)
    step2_text = tokenizer.decode([int(step2_token)])
    print(f"Step 2: token={step2_token}, text='{step2_text}', logit={step2_last[step2_token]:.4f}")

    print(f"\n=== Summary: Step 0={step0_token}('{step0_text}'), Step 1={step1_token}('{step1_text}'), Step 2={step2_token}('{step2_text}') ===")

if __name__ == "__main__":
    main()
