#!/usr/bin/env python3
"""Inspect the decoder ONNX model to find the causal mask computation."""
import os, sys
try:
    import onnx
    import onnxruntime as ort
    import numpy as np
except ImportError:
    print("pip install onnx onnxruntime numpy")
    sys.exit(1)

CACHE = os.path.expanduser("~/.cache/dl4j-vlm-models")
DECODER = os.path.join(CACHE, "smoldocling-decoder.onnx")

model = onnx.load(DECODER)
graph = model.graph

# Find nodes related to causal mask computation
# Look for Where, Less, Greater, comparison ops near the beginning
print("=== Searching for causal mask / attention mask related nodes ===")
for i, node in enumerate(graph.node):
    name = node.name or f"node_{i}"
    op_type = node.op_type
    # Look for comparison and mask ops
    if op_type in ['Where', 'Less', 'LessOrEqual', 'Greater', 'GreaterOrEqual',
                   'Equal', 'Not', 'And', 'Or', 'Unsqueeze', 'Expand',
                   'Range', 'ConstantOfShape']:
        # Check if near attention mask or causal mask path
        inputs_str = ', '.join(node.input)
        outputs_str = ', '.join(node.output)
        if any('mask' in x.lower() or 'causal' in x.lower() or
               'attention' in x.lower() or 'where' in name.lower() or
               'attn' in x.lower() for x in list(node.input) + list(node.output) + [name]):
            print(f"  [{i}] {op_type} '{name}': {inputs_str} -> {outputs_str}")

print("\n=== All nodes in first 150 (before first attention layer) ===")
for i, node in enumerate(graph.node[:150]):
    name = node.name or f"node_{i}"
    op_type = node.op_type
    inputs_str = ', '.join(node.input[:4])
    if len(node.input) > 4:
        inputs_str += f"... (+{len(node.input)-4} more)"
    outputs_str = ', '.join(node.output[:2])
    print(f"  [{i:3d}] {op_type:20s} '{name}' -> [{inputs_str}] -> [{outputs_str}]")

# Also find the "attn" related nodes in the first layer
print("\n=== Nodes in first attention layer (layers.0/attn) ===")
for i, node in enumerate(graph.node):
    name = node.name or f"node_{i}"
    if 'layers.0/attn' in name or 'layers.0/self_attn' in name:
        inputs_str = ', '.join(node.input[:3])
        outputs_str = ', '.join(node.output[:2])
        print(f"  [{i:3d}] {node.op_type:20s} '{name}' -> [{inputs_str}] -> [{outputs_str}]")
