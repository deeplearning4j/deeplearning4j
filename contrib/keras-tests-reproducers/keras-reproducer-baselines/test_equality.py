import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2DTranspose

nIn = 2
nOut = 3
kH = 2
kW = 2

mb = 2
imgH = 8
imgW = 8

# Define the input arrays
wArr = np.linspace(0, 1, kH * kW * nIn * nOut, dtype=np.float32).reshape(kH, kW, nIn, nOut)
bArr = np.linspace(0, 1, nOut, dtype=np.float32)
inArr = np.linspace(0, 1, mb * imgH * imgW * nIn, dtype=np.float32).reshape(mb, imgH, imgW, nIn)

# Define input tensor
inputs = Input(shape=(imgH, imgW, nIn), batch_size=mb)

# Define Conv2DTranspose layer
deconv2d = Conv2DTranspose(
    filters=nOut,
    kernel_size=(kH, kW),
    strides=(1, 1),
    padding='valid',
    output_padding=None,
    dilation_rate=(1, 1),
    data_format='channels_last',
    use_bias=False,
    kernel_initializer=tf.constant_initializer(wArr),
    bias_initializer=tf.constant_initializer(bArr),
)(inputs)

# Define loss function
outputs = tf.math.reduce_std(tf.math.tanh(deconv2d), keepdims=True)
model = Model(inputs=inputs, outputs=outputs)

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
    tape.watch(model.layers[1].kernel)
    x_tensor = tf.convert_to_tensor(inArr, dtype=tf.float32)
    outputs = model(x_tensor, training=False)
    kernel_grad = tape.gradient(outputs, model.layers[1].kernel)
    backward_pass_outputs[model.layers[1].name] = {
        "kernel_grad": kernel_grad.numpy(),
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