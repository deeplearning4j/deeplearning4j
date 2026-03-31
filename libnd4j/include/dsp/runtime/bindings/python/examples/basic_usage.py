"""Minimal SDX Runtime usage example."""

import numpy as np
from sdx_runtime import SdxRuntime

# Create runtime (auto-detects library from SDK layout)
runtime = SdxRuntime()
print(f"ABI version: {runtime.abi_version()}")

# Load a compiled model bundle
model = runtime.load_model("path/to/model.sdx")

# Create execution context
ctx = model.create_context()

# Prepare input as a numpy array
input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

# Run inference
outputs = ctx.run([input_data])
print(f"Output shape: {outputs[0].shape}")
print(f"Output values: {outputs[0]}")

# Cleanup
ctx.close()
model.close()
runtime.close()
