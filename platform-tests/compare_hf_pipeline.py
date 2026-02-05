#!/usr/bin/env python3
"""Run the actual HuggingFace transformers pipeline for SmolDocling
to see what it does differently from our raw ONNX approach."""

import torch
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import os

PDF_PATH = "pathfinder-mythic.pdf"
PAGE_NUM = 10

def render_pdf_page(pdf_path, page_num, dpi=150):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def run_pipeline(processor, model, img, prompt_text, label="default"):
    print(f"\n{'='*60}")
    print(f"=== Pipeline run: {label} ===")
    print(f"{'='*60}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    print(f"Chat template ({len(text)} chars): {repr(text[:200])}...{repr(text[-100:])}")

    inputs = processor(text=[text], images=[img], return_tensors="pt")

    print(f"pixel_values: shape={inputs['pixel_values'].shape}")
    print(f"pixel_attention_mask: shape={inputs['pixel_attention_mask'].shape}")
    print(f"input_ids: shape={inputs['input_ids'].shape}")

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    image_count = (inputs['input_ids'] == image_token_id).sum().item()
    num_frames = inputs['pixel_values'].shape[1]
    print(f"Frames: {num_frames}, <image> tokens: {image_count}, tokens_per_frame: {image_count // num_frames if num_frames > 0 else 0}")

    # Run generation
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )

    new_tokens = generated[0][inputs['input_ids'].shape[1]:]
    new_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=False)
    print(f"New token ids: {new_tokens.tolist()}")
    print(f"New text: '{new_text}'")

    # Also get logits for first token
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        last_logits = logits[0, -1, :]
        top5_ids = torch.argsort(last_logits, descending=True)[:5]
        print(f"\nPrefill logits shape: {logits.shape}")
        print(f"Last-position top-5:")
        for tid in top5_ids:
            text_tok = processor.tokenizer.decode([tid.item()])
            print(f"  id={tid.item()}, logit={last_logits[tid].item():.4f}, text='{text_tok}'")
        print(f"Token 216 (space): logit={last_logits[216].item():.4f}")
        print(f"Token 49229 (<doctag>): logit={last_logits[49229].item():.4f}")

    # Trace internals
    with torch.no_grad():
        # Get image features
        pixel_values = inputs['pixel_values']
        pixel_attention_mask = inputs.get('pixel_attention_mask', None)

        nb_values_per_image = pixel_values.shape[1]
        real_batch = pixel_values.shape[0]
        pixel_values_flat = pixel_values.reshape(
            real_batch * nb_values_per_image, *pixel_values.shape[2:]
        )

        if pixel_attention_mask is not None:
            pixel_attention_mask_flat = pixel_attention_mask.reshape(
                real_batch * nb_values_per_image, *pixel_attention_mask.shape[2:]
            )
        else:
            pixel_attention_mask_flat = None

        # Raw vision encoder output
        raw_out = model.model.vision_model(
            pixel_values=pixel_values_flat,
            patch_attention_mask=pixel_attention_mask_flat,
        )
        raw_hidden = raw_out.last_hidden_state
        print(f"\nRaw vision hidden: shape={raw_hidden.shape}, min={raw_hidden.min():.4f}, max={raw_hidden.max():.4f}")

        # After connector
        connected = model.model.connector(raw_hidden)
        print(f"After connector: shape={connected.shape}, min={connected.min():.4f}, max={connected.max():.4f}")

        # Reshape to image features per batch
        image_features = connected.reshape(real_batch, nb_values_per_image * connected.shape[1], connected.shape[2])
        print(f"Image features (reshaped): shape={image_features.shape}")

        # Save for comparison
        np.save(f"/tmp/hf_{label}_image_features.npy", image_features.numpy())

        # Get text embeddings
        input_ids = inputs['input_ids']
        text_embeddings = model.model.text_model.get_input_embeddings()(input_ids)
        print(f"Text embeddings: shape={text_embeddings.shape}, min={text_embeddings.min():.4f}, max={text_embeddings.max():.4f}")

        # Show how HF merges - look at inputs_embeds_with_image
        # The merge happens in model.model._merge_input_ids_with_image_features
        # Let's replicate it
        merged = text_embeddings.clone()
        image_positions = (input_ids == image_token_id).squeeze(0)
        n_image_positions = image_positions.sum().item()
        print(f"Image positions count: {n_image_positions}, image features tokens: {image_features.shape[1]}")

        if n_image_positions == image_features.shape[1]:
            merged[0, image_positions] = image_features[0]
            print(f"Merged embeddings: min={merged.min():.4f}, max={merged.max():.4f}")
        else:
            print(f"MISMATCH: {n_image_positions} positions vs {image_features.shape[1]} features")

    return new_text

def main():
    print("=== HuggingFace Transformers Pipeline Reference ===")

    from transformers import AutoProcessor, Idefics3ForConditionalGeneration

    model_name = "ds4sd/SmolDocling-256M-preview"
    print(f"Loading model: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name)
    model = Idefics3ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    print(f"Model type: {type(model).__name__}")
    print(f"Connector type: {type(model.model.connector).__name__}")

    # Check connector architecture
    print(f"\nConnector details:")
    for name, param in model.model.connector.named_parameters():
        print(f"  {name}: shape={param.shape}")

    img = render_pdf_page(PDF_PATH, PAGE_NUM)
    print(f"Rendered image: {img.size}")
    prompt_text = "Convert this page to docling."

    # Run with default settings (no max_tiles limit)
    run_pipeline(processor, model, img, prompt_text, "default_17frames")

    # Run with max_image_tiles=4 to match our Java test
    print("\n\n" + "="*80)
    print("Now testing with max_image_tiles=4 (matching Java test)")
    print("="*80)
    processor4 = AutoProcessor.from_pretrained(model_name, max_image_tiles=4)
    run_pipeline(processor4, model, img, prompt_text, "4tiles")

    print("\n=== HuggingFace Reference Complete ===")

if __name__ == "__main__":
    main()
