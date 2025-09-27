import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

nIn = 2
nOut = 3
kH = 2
kW = 2

mb = 2
imgH = 8
imgW = 8

# Define the input arrays
wArr = np.linspace(1, kH * kW * nOut * nIn, num=kH * kW * nOut * nIn).reshape(kH, kW, nIn, nOut)
bArr = np.linspace(1, nOut, num=nOut)
inArr = np.linspace(1, mb * imgH * imgW * nIn, num=mb * imgH * imgW * nIn).reshape(mb, imgH, imgW, nIn)

# Create the model
model = keras.Sequential([
    layers.Input(shape=(imgH, imgW, nIn)),
    layers.Conv2DTranspose(
        filters=nOut,
        kernel_size=(kH, kW),
        strides=(1, 1),
        padding='valid',
        output_padding=None,
        dilation_rate=(1, 1),
        activation='tanh',
        kernel_initializer=keras.initializers.constant(wArr),
        bias_initializer=keras.initializers.constant(bArr)
    )
])

# Print the model summary
model.summary()

# Compute forward pass
forward_pass_outputs = {}
for layer in model.layers:
    layer_output = layer(layer.input)
    submodel = Model(model.inputs, layer.output)
    results = submodel.predict(inArr)
    forward_pass_outputs[layer.name] = results

# Compute backward pass
backward_pass_outputs = {}
with tf.GradientTape(persistent=True) as tape:
    tape.watch(model.layers[0].kernel)
    tape.watch(model.layers[0].bias)
    x_tensor = tf.convert_to_tensor(inArr, dtype=tf.float32)
    outputs = model(x_tensor, training=False)
    kernel_grad = tape.gradient(outputs, model.layers[0].kernel)
    bias_grad = tape.gradient(outputs, model.layers[0].bias)
    backward_pass_outputs[model.layers[0].name] = {
        "kernel_grad": kernel_grad.numpy(),
        "bias_grad": bias_grad.numpy()
    }

# Print the forward pass outputs
print("Forward Pass Outputs:")
for layer_name, output in forward_pass_outputs.items():
    print(f"Layer: {layer_name}")
    print(f"Output Shape: {output.shape}")
    print(f"Output Values: {output}")
    print()

# Print the backward pass outputs
print("Backward Pass Outputs:")
for layer_name, grads in backward_pass_outputs.items():
    print(f"Layer: {layer_name}")
    print(f"Kernel Gradient Shape: {grads['kernel_grad'].shape}")
    print(f"Kernel Gradient Values: {grads['kernel_grad']}")
    print(f"Bias Gradient Shape: {grads['bias_grad'].shape}")
    print(f"Bias Gradient Values: {grads['bias_grad']}")
    print()