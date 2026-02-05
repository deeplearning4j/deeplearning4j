#!/usr/bin/env python3
"""Run the actual HuggingFace transformers pipeline for SmolDocling to see generation results."""

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

def main():
    print("=== HuggingFace Generate Reference ===")

    from transformers import AutoProcessor, Idefics3ForConditionalGeneration

    model_name = "ds4sd/SmolDocling-256M-preview"
    print(f"Loading model: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name)
    model = Idefics3ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    # Render PDF page
    img = render_pdf_page(PDF_PATH, PAGE_NUM)
    print(f"Rendered image: {img.size}")

    # Use the processor
    prompt_text = "Convert this page to docling."
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
    print(f"Chat template: {repr(text)}")

    inputs = processor(text=[text], images=[img], return_tensors="pt")
    print(f"pixel_values: shape={inputs['pixel_values'].shape}")
    print(f"input_ids: shape={inputs['input_ids'].shape}")

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    image_count = (inputs['input_ids'] == image_token_id).sum().item()
    print(f"<image> tokens: {image_count}")

    # Generate with the model
    print("\n=== Generating (default, 17 frames) ===")
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

    # Now try with max_tiles=4 to match our Java test
    print("\n=== Generating (max_tiles=4, 5 frames) ===")
    # Use processor with max_image_tiles=4
    # The Idefics3ImageProcessor accepts this as a parameter
    processor4 = AutoProcessor.from_pretrained(model_name)
    processor4.image_processor.max_image_tiles = 4

    inputs4 = processor4(text=[text], images=[img], return_tensors="pt")
    print(f"pixel_values: shape={inputs4['pixel_values'].shape}")
    print(f"input_ids: shape={inputs4['input_ids'].shape}")
    image_count4 = (inputs4['input_ids'] == image_token_id).sum().item()
    print(f"<image> tokens: {image_count4}")

    with torch.no_grad():
        generated4 = model.generate(
            **inputs4,
            max_new_tokens=20,
            do_sample=False,
        )

    new_tokens4 = generated4[0][inputs4['input_ids'].shape[1]:]
    new_text4 = processor4.tokenizer.decode(new_tokens4, skip_special_tokens=False)
    print(f"New token ids: {new_tokens4.tolist()}")
    print(f"New text: '{new_text4}'")

    # Also trace intermediate values for the 4-tile case
    print("\n=== Tracing 4-tile internals ===")
    with torch.no_grad():
        # Get inputs_embeds the way the model does it
        pixel_values = inputs4['pixel_values']
        pixel_attention_mask = inputs4.get('pixel_attention_mask', None)
        input_ids = inputs4['input_ids']
        attention_mask = inputs4['attention_mask']

        # Step 1: Vision encoder
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values_flat = pixel_values.reshape(batch_size * num_images, num_channels, height, width)

        if pixel_attention_mask is not None:
            pixel_attention_mask_flat = pixel_attention_mask.reshape(
                batch_size * num_images, height, width
            )
        else:
            pixel_attention_mask_flat = None

        vision_outputs = model.model.vision_model(
            pixel_values=pixel_values_flat,
            patch_attention_mask=pixel_attention_mask_flat,
        )
        raw_hidden = vision_outputs.last_hidden_state
        print(f"Raw vision hidden: shape={raw_hidden.shape}, min={raw_hidden.min():.4f}, max={raw_hidden.max():.4f}")

        # Step 2: Connector (modality projection)
        image_features = model.model.connector(raw_hidden)
        image_features = image_features.reshape(batch_size, num_images * image_features.shape[1], -1)
        print(f"After connector: shape={image_features.shape}, min={image_features.min():.4f}, max={image_features.max():.4f}, mean={image_features.mean():.4f}")

        # Save for comparison
        np.save("/tmp/hf_image_features_4tiles.npy", image_features.numpy())

        # Step 3: Text embeddings
        text_embeddings = model.model.text_model.get_input_embeddings()(input_ids)
        print(f"Text embeddings: shape={text_embeddings.shape}, min={text_embeddings.min():.4f}, max={text_embeddings.max():.4f}, mean={text_embeddings.mean():.4f}")

        # Step 4: Merge
        inputs_embeds = text_embeddings.clone()
        image_mask = input_ids == image_token_id
        num_image_tokens = image_mask.sum().item()
        print(f"Image token positions: {num_image_tokens}")
        print(f"Image features available: {image_features.shape[1]}")

        # The model's merge logic
        inputs_embeds[image_mask] = image_features.reshape(-1, image_features.shape[-1]).to(inputs_embeds.dtype)
        print(f"Merged embeddings: shape={inputs_embeds.shape}, min={inputs_embeds.min():.4f}, max={inputs_embeds.max():.4f}")

        # Save merged embeddings for comparison
        np.save("/tmp/hf_merged_embeddings_4tiles.npy", inputs_embeds.numpy())

        # Step 5: Run decoder to get prefill logits
        outputs = model.model.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        logits = model.lm_head(hidden_states)
        print(f"Logits: shape={logits.shape}, min={logits.min():.4f}, max={logits.max():.4f}")

        last_logits = logits[0, -1, :].numpy()
        top5_ids = np.argsort(last_logits)[-5:][::-1]
        print("\nTop-5 tokens at last position:")
        for tid in top5_ids:
            text_tok = processor.tokenizer.decode([int(tid)])
            print(f"  id={tid}, logit={last_logits[tid]:.4f}, text='{text_tok}'")

        print(f"\nToken 216 (space): logit={last_logits[216]:.4f}")
        print(f"Token 49229 (<doctag>): logit={last_logits[49229]:.4f}")

    print("\n=== Done ===")

if __name__ == "__main__":
    main()
