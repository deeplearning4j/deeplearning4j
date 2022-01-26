from transformers import AutoModelForSeq2SeqLM

from omnihub.frameworks.huggingface import HuggingFaceModelHub
from omnihub.frameworks.keras import KerasModelHub
from omnihub.frameworks.onnx import OnnxModelHub
from omnihub.frameworks.pytorch import PytorchModelHub
from omnihub.frameworks.tensorflow import TensorflowModelHub

keras_model_hub = KerasModelHub()
keras_urls = [
    #'vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #          'vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5',
    # bug in downloader? Seems to be stalled at the last few bytes, skipping for now
    #'vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5',
    #'vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
    'resnet50/notop',
    'resnet50/top',
    'resnet101/notop',
    'resnet101/top',
    'resnet152/notop',
    'resnet152/top',
    'resnet50v2/notop',
    'resnet50v2/top',
    'resnet101v2/notop',
    'resnet101v2/top',
    'resnet152v2/notop',
    'resnet152v2/top',
    'densenet121/notop',
    'densenet121/top',
    'densenet169/notop',
    'densenet169/top',
    'densenet201/notop',
    'densenet201/top',
    'inceptionresnetv2/notop',
    'inceptionresnetv2/top',
    'mobilenet/notop',
    'mobilenet/top',
    'mobilenetv2/notop',
    'mobilenetv2/top',
    #'mobilenetv3/notop',
    #'mobilenetv3/top',
    'nasnet/notop',
    'nasnet/top',
    'nasnet_mobile/notop',
    'nasnet_mobile/top',
    'xception/notop',
    'xception/top',
]
for i in range(0,8):
    keras_urls.append(f'efficientnetb{i}')



#for url in keras_urls:
#    keras_model_hub.download_model(url)
onnx_model_hub = OnnxModelHub()
onnx_urls = ['vision/body_analysis/age_gender/models/age_googlenet.onnx',
             'vision/body_analysis/age_gender/models/gender_googlenet.onnx',
             'vision/body_analysis/age_gender/models/vgg_ilsvrc_16_age_chalearn_iccv2015.onnx',
             'vision/body_analysis/age_gender/models/vgg_ilsvrc_16_age_imdb_wiki.onnx',
             'vision/body_analysis/age_gender/models/vgg_ilsvrc_16_gender_imdb_wiki.onnx',
             'vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx',
             'vision/body_analysis/emotion_ferplus/model/emotion-ferplus-2.onnx',
             'vision/body_analysis/emotion_ferplus/model/emotion-ferplus-7.onnx',
             'vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx',
             'vision/body_analysis/ultraface/models/version-RFB-320.onnx',
             'vision/body_analysis/ultraface/models/version-RFB-640.onnx',
             'vision/classification/alexnet/model/bvlcalexnet-12-int8.onnx',
             'vision/classification/alexnet/model/bvlcalexnet-12.onnx',
             'vision/classification/caffenet/model/caffenet-12-int8.onnx',
             'vision/classification/caffenet/model/caffenet-12.onnx',
             'vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx'
             ]
#for url in onnx_urls:
#    onnx_model_hub.download_model(url)
tensorflow_model_hub = TensorflowModelHub()
tensorflow_urls = ['emilutz/vgg19-block4-conv2-unpooling-decoder/1']
for url in tensorflow_urls:
    tensorflow_model_hub.download_model(url)
pytorch_model_hub = PytorchModelHub()
pytorch_urls = ['resnet18']
for url in pytorch_urls:
    pytorch_model_hub.download_model(url)
huggingface_model_hub = HuggingFaceModelHub()
frameworks = ['pytorch']
huggingface_urls =  [ 'gpt2','bert-base-uncased','t5-base','bert-base-chinese','google/electra-small-discriminator','facebook/wav2vec2-base-960h','facebook/bart-large-cnn']
huggingface_urls = ['openai/clip-vit-base-patch32']
for url in huggingface_urls:
    for framework_name in frameworks:
        huggingface_model_hub.download_model(url,framework_name=framework_name)
