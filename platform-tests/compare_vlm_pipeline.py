#!/usr/bin/env python3
"""Compare VLM pipeline outputs between Python (onnxruntime) and Java (SameDiff).
Runs the same ONNX models with the same image to find where divergence occurs."""

import numpy as np
import onnxruntime as ort
from PIL import Image
import fitz  # PyMuPDF
import os
import sys
from tokenizers import Tokenizer

MODEL_DIR = os.path.expanduser("~/.cache/dl4j-vlm-models")
PDF_PATH = "pathfinder-mythic.pdf"
PAGE_NUM = 10  # 0-indexed in PyMuPDF
TARGET_SIZE = 512
MAX_TILES = 4

def load_tokenizer():
    tok_path = os.path.join(MODEL_DIR, "smoldocling-tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)
    return tokenizer

def render_pdf_page(pdf_path, page_num, dpi=150):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def resize_for_tiling(img, target_size):
    """Resize image so dimensions are multiples of target_size."""
    w, h = img.size
    # Scale to fit: max dimension gets scaled to be a multiple of target_size
    aspect = w / h
    # Make dimensions multiples of target_size
    new_w = max(target_size, ((w + target_size - 1) // target_size) * target_size)
    new_h = max(target_size, ((h + target_size - 1) // target_size) * target_size)
    # But respect the original aspect ratio approximately
    if aspect >= 1:
        new_h = max(target_size, round(new_w / aspect / target_size) * target_size)
    else:
        new_w = max(target_size, round(new_h * aspect / target_size) * target_size)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    return img_resized

def split_image(img, target_size, max_tiles):
    """Split image into tiles + global frame."""
    w, h = img.size
    cols = w // target_size
    rows = h // target_size

    # Reduce grid to fit max_tiles
    while rows * cols > max_tiles:
        if rows >= cols:
            rows -= 1
        else:
            cols -= 1

    print(f"Grid: {rows}x{cols} ({rows*cols} tiles)")

    # Compute optimal tile size
    tile_w = w // cols
    tile_h = h // rows

    frames = []
    regions = []  # (actual_w, actual_h) for each frame

    for r in range(rows):
        for c in range(cols):
            x0 = c * tile_w
            y0 = r * tile_h
            x1 = min(x0 + tile_w, w)
            y1 = min(y0 + tile_h, h)
            tile = img.crop((x0, y0, x1, y1)).resize((target_size, target_size), Image.LANCZOS)
            frames.append(tile)
            regions.append((x1 - x0, y1 - y0))

    # Global frame
    global_frame = img.resize((target_size, target_size), Image.LANCZOS)
    frames.append(global_frame)
    regions.append((w, h))

    return frames, regions, rows, cols

def preprocess_frame(frame, target_size, normalize=True):
    """Preprocess a single frame for the vision encoder."""
    arr = np.array(frame).astype(np.float32) / 255.0

    if normalize:
        # SigLIP normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        arr = (arr - 0.5) / 0.5

    # HWC -> CHW
    arr = arr.transpose(2, 0, 1)
    # Add batch and frame dims: [1, 1, C, H, W]
    arr = arr.reshape(1, 1, 3, target_size, target_size)
    return arr

def create_pixel_attention_mask(content_w, content_h, target_size):
    """Create pixel attention mask for a frame."""
    # Scale content dimensions to target_size
    scale_x = target_size / content_w if content_w > 0 else 1.0
    scale_y = target_size / content_h if content_h > 0 else 1.0
    # For tiles that perfectly fill the space, mask is all ones
    mask = np.ones((1, 1, target_size, target_size), dtype=np.bool_)
    return mask

def build_image_prompt(num_rows, num_cols, seq_len_per_image):
    """Build the image prompt string matching HuggingFace format."""
    fake = "<fake_token_around_image>"
    image = "<image>"
    glob = "<global-img>"

    sb = []
    for r in range(1, num_rows + 1):
        for c in range(1, num_cols + 1):
            sb.append(fake)
            sb.append(f"<row_{r}_col_{c}>")
            sb.append(image * seq_len_per_image)
        sb.append("\n")

    sb.append("\n")
    sb.append(fake)
    sb.append(glob)
    sb.append(image * seq_len_per_image)
    sb.append(fake)

    return "".join(sb)

def main():
    print("=== Python VLM Pipeline Reference ===")

    # Load tokenizer
    tokenizer = load_tokenizer()
    print(f"Tokenizer loaded, vocab_size={tokenizer.get_vocab_size()}")

    # Get special token IDs
    image_token_id = tokenizer.token_to_id("<image>")
    print(f"<image> token id: {image_token_id}")

    # Load ONNX models
    print("Loading ONNX models...")
    vision_sess = ort.InferenceSession(
        os.path.join(MODEL_DIR, "smoldocling-vision-encoder.onnx"),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    decoder_sess = ort.InferenceSession(
        os.path.join(MODEL_DIR, "smoldocling-decoder.onnx"),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    embed_sess = ort.InferenceSession(
        os.path.join(MODEL_DIR, "smoldocling-embed-tokens.onnx"),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    print("Vision encoder inputs:", [i.name for i in vision_sess.get_inputs()])
    print("Vision encoder outputs:", [o.name for o in vision_sess.get_outputs()])
    print("Decoder inputs:", [i.name for i in decoder_sess.get_inputs()])
    print("Decoder outputs:", [o.name for o in decoder_sess.get_outputs()])
    print("Embed inputs:", [i.name for i in embed_sess.get_inputs()])
    print("Embed outputs:", [o.name for o in embed_sess.get_outputs()])

    # Render PDF page
    print(f"\nRendering PDF page {PAGE_NUM}...")
    img = render_pdf_page(PDF_PATH, PAGE_NUM)
    print(f"Rendered image: {img.size}")

    # Resize for tiling
    img_resized = resize_for_tiling(img, TARGET_SIZE)
    print(f"Resized for tiling: {img_resized.size}")

    # Split into tiles
    frames, regions, num_rows, num_cols = split_image(img_resized, TARGET_SIZE, MAX_TILES)
    print(f"Total frames: {len(frames)} ({num_rows*num_cols} tiles + 1 global)")

    # Process each frame through vision encoder
    frame_embeddings = []
    for i, frame in enumerate(frames):
        pixel_values = preprocess_frame(frame, TARGET_SIZE, normalize=True)
        pixel_mask = create_pixel_attention_mask(regions[i][0], regions[i][1], TARGET_SIZE)

        if i == 0:
            print(f"\nFrame {i} pixel_values: shape={pixel_values.shape}, min={pixel_values.min():.4f}, max={pixel_values.max():.4f}, mean={pixel_values.mean():.4f}")

        outputs = vision_sess.run(None, {
            'pixel_values': pixel_values,
            'pixel_attention_mask': pixel_mask
        })

        emb = outputs[0]  # image_features
        if i == 0:
            print(f"Frame {i} output: shape={emb.shape}, min={emb.min():.4f}, max={emb.max():.4f}, mean={emb.mean():.4f}")
        frame_embeddings.append(emb)

    # Concatenate frame embeddings
    vision_embeddings = np.concatenate(frame_embeddings, axis=1)
    print(f"\nCombined vision embeddings: shape={vision_embeddings.shape}")

    # Build prompt
    image_seq_len = frame_embeddings[0].shape[1]
    image_prompt = build_image_prompt(num_rows, num_cols, image_seq_len)
    prompt_text = "Convert this page to docling."
    chat_prompt = f"<|im_start|>User:{image_prompt}{prompt_text}<end_of_utterance>\nAssistant:"
    print(f"Chat prompt length: {len(chat_prompt)} chars")

    # Tokenize
    encoding = tokenizer.encode(chat_prompt)
    prompt_token_ids = encoding.ids
    print(f"Tokenized prompt: {len(prompt_token_ids)} tokens")

    image_token_count = sum(1 for t in prompt_token_ids if t == image_token_id)
    print(f"<image> tokens in prompt: {image_token_count}, vision tokens: {vision_embeddings.shape[1]}")

    # Print first/last tokens
    print("First 15 tokens:", [(tid, tokenizer.decode([tid])) for tid in prompt_token_ids[:15]])
    print("Last 10 tokens:", [(tid, tokenizer.decode([tid])) for tid in prompt_token_ids[-10:]])

    # Get text embeddings
    input_ids = np.array([prompt_token_ids], dtype=np.int64)
    embed_outputs = embed_sess.run(None, {'input_ids': input_ids})
    text_embeddings = embed_outputs[0]
    print(f"\nText embeddings: shape={text_embeddings.shape}")

    # Merge vision and text embeddings
    seq_len = len(prompt_token_ids)
    hidden_dim = vision_embeddings.shape[2]
    merged = text_embeddings.copy()

    fill_idx = 0
    vision_flat = vision_embeddings.reshape(-1, hidden_dim)
    for pos in range(seq_len):
        if prompt_token_ids[pos] == image_token_id and fill_idx < vision_flat.shape[0]:
            merged[0, pos, :] = vision_flat[fill_idx]
            fill_idx += 1

    print(f"Merged embeddings: shape={merged.shape}, min={merged.min():.4f}, max={merged.max():.4f}")
    print(f"Filled {fill_idx} image token positions")

    # Save merged embeddings for comparison
    np.save("/tmp/python_merged_embeddings.npy", merged)
    print("Saved merged embeddings to /tmp/python_merged_embeddings.npy")

    # Run decoder (prefill)
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

    # Create empty KV caches
    num_layers = 30
    num_kv_heads = 3
    head_dim = 64

    decoder_inputs = {
        'inputs_embeds': merged.astype(np.float32),
        'attention_mask': attention_mask,
        'position_ids': position_ids,
    }

    for i in range(num_layers):
        decoder_inputs[f'past_key_values.{i}.key'] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float32)
        decoder_inputs[f'past_key_values.{i}.value'] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float32)

    print("\nRunning decoder (prefill)...")
    decoder_outputs = decoder_sess.run(None, decoder_inputs)

    # First output is logits
    logits = decoder_outputs[0]
    print(f"Logits: shape={logits.shape}, min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")

    # Get last-position logits
    last_logits = logits[0, -1, :]
    print(f"Last-position logits: min={last_logits.min():.4f}, max={last_logits.max():.4f}")

    # Top-5 tokens
    top5_ids = np.argsort(last_logits)[-5:][::-1]
    print("\nTop-5 tokens at last position:")
    for tid in top5_ids:
        text = tokenizer.decode([int(tid)])
        print(f"  id={tid}, logit={last_logits[tid]:.4f}, text='{text}'")

    # Check specific tokens
    print(f"\nToken 216 (space): logit={last_logits[216]:.4f}")
    print(f"Token 49229 (<doctag>): logit={last_logits[49229]:.4f}")

    # Argmax sampling
    next_token = np.argmax(last_logits)
    next_text = tokenizer.decode([int(next_token)])
    print(f"\nArgmax next token: id={next_token}, text='{next_text}'")

    # Save logits for comparison
    np.save("/tmp/python_prefill_logits.npy", logits)
    print("Saved logits to /tmp/python_prefill_logits.npy")

    # Also print the first 20 logits at last position for direct comparison
    print(f"\nLogits[0,-1,0:20]: {last_logits[:20]}")

    print("\n=== Python Reference Complete ===")

if __name__ == "__main__":
    main()
