from omnihub.frameworks.huggingface import HuggingFaceModelHub
from omnihub.frameworks.keras import KerasModelHub
from omnihub.frameworks.onnx import OnnxModelHub
from omnihub.frameworks.pytorch import PytorchModelHub
from omnihub.frameworks.tensorflow import TensorflowModelHub

keras_model_hub = KerasModelHub()
keras_urls = ['vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5']
for url in keras_urls:
    keras_model_hub.download_model(url)
onnx_model_hub = OnnxModelHub()
onnx_urls = ['vision/body_analysis/age_gender/models/age_googlenet.onnx']
for url in onnx_urls:
    onnx_model_hub.download_model(url)
tensorflow_model_hub = TensorflowModelHub()
tensorflow_urls = ['emilutz/vgg19-block4-conv2-unpooling-decoder/1']
for url in tensorflow_urls:
    tensorflow_model_hub.download_model(url)
pytorch_model_hub = PytorchModelHub()
pytorch_urls = ['resnet18']
for url in pytorch_urls:
    pytorch_model_hub.download_model(url)
huggingface_model_hub = HuggingFaceModelHub()
huggingface_urls =  { huggingface_model_hub.resolve_url('gpt2', 'tf_model.h5'): 'keras' }
for url,framework_name in huggingface_urls.items():
    huggingface_model_hub.download_model(url,framework_name=framework_name)
