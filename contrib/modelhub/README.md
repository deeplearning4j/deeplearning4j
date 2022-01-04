Omnihub
--------------------------
Simple downloading and conversion of pretrained models

###Setup
```bash
pip install -r requirements.txt
python setup.py install 
```

###Basic Usage
See the [unit tests](src/tests/omnihub/test_frameworks.py) for basic usage
Simple example:

```python
from omnihub import OnnxModelHub

keras_model_hub = KerasModelHub()
model_path = keras_model_hub.download_model('vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
keras_model_hub.stage_model(model_path, 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
```

This will download a model using keras applications and put it in:
```bash
$HOME/.model_hub/keras/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
```

The basic idea is that each framework has its "model hub" which knows how interact
with and pre process models from different frameworks. The goal is to encapsulate common
steps per framework such as freezing/unfreezing, downloading of models.



###Background
-------------

An SDK for interacting with various model zoos across different frameworks.
Model hub handles downloading and initializing models from different model zoos
handling conversion to standalone files. Various complexities
across different frameworks exist for making deployable or finetunable model files.

Finetuning a model involves usually:
1. Unfreezing a model(converting constants to variables)
2. Customizing a model (adding a new objective plus other layers on the end)

Other steps may optionally exist but these are the 2 main ones. Doing this
across different frameworks varies in complexity.

Making a model deployable typically involves:
1. freezing a model (convert trainable parameters to frozen constants)
2. Optimizing a model (quantizing it, changing the data type, removing extra operations to reduce model size,..)

These are 2 common workflows that require reusing an existing model file
produced by a framework such as tensorflow or pytorch.
All of these still come with a fair amount of friction that involves 
1 off tutorials and copy and paste praying it will work.

