import os
from omnihub.model_hub import omnihub_dir, ModelHub
import torch
from torchvision import models
import numpy as np

framework_name = 'pytorch'
framework_dir = os.path.join(omnihub_dir, framework_name)
BASE_URL = 'https://s3.amazonaws.com/pytorch/models'

# models with default 224 x 224 height,width
MODEL_224_DEFAULTS = ['resnet18', 'vgg16', 'shufflenet_v2_x1_0', 'resnext50_32x4d', 'wide_resnet50_2', 'mnasnet1_0']
# models with default 256 x 256 height,width
MODEL_256_DEFAULTS = ['alexnet', 'squeezenet1_0', 'densenet161', 'googlenet', 'inception_v3', 'fasterrcnn', 'ssd',
                      'retinanet', 'maskrcnn', 'keypointrcnn']
# models in pytorch's model.detection
detection_models = ['fasterrcnn', 'ssd', 'retinanet', 'maskrcnn', 'keypointrcnn', 'retinanet']
# misc defaults and base dictionary for storing heights,widths
MODEL_DEFAULTS = {
    'mobilenet_v2': {
        'height': 32,
        'width': 32
    },
    'mobilenet_v3_large': {
        'height': 320,
        'width': 320
    },
    'mobilenet_v3_small': {
        'height': 320,
        'width': 320
    },
    'retinanet': {
        'height': 512,
        'width': 512
    },

}

# efficient_b0 to 7 has height,width default 256 x 256
for i in range(0, 8):
    MODEL_256_DEFAULTS.append(f'efficientnet_b{i}')

#regnet_x/y_sizes has default height,width 256 x 256
regnet_suffix_sizes = ['400mf', '800mf', '1_6gf', '3_2gf', '8gf', '16gf', '32gf']
for suffix in ['x', 'y']:
    for size in regnet_suffix_sizes:
        MODEL_256_DEFAULTS.append(f'regnet_{suffix}_{size}')

for model_224 in MODEL_224_DEFAULTS:
    MODEL_DEFAULTS[model_224] = {
        'height': 224,
        'width': 224
    }

for model_256 in MODEL_256_DEFAULTS:
    MODEL_DEFAULTS[model_256] = {
        'height': 256,
        'width': 256
    }


class PytorchModelHub(ModelHub):
    def __init__(self):
        super().__init__(framework_name, BASE_URL)

    def download_model(self, model_path, **kwargs) -> str:
        model = None
        height = kwargs.get('height', MODEL_DEFAULTS[model_path]['height'])
        width = kwargs.get('width', MODEL_DEFAULTS[model_path]['width'])
        x = torch.from_numpy(np.ones((1, 3, height, width), dtype=np.float32))
        if model_path in detection_models:
            model = models.detection[model_path](pretrained=True, **kwargs)
        else:
            model = models.__dict__[model_path](pretrained=True, **kwargs)
        torch.onnx.export(model,
                          x,
                          f'{framework_dir}/{model_path}.onnx',
                          export_params=True,
                          do_constant_folding=False,
                          opset_version=13,
                          **kwargs)

    def stage_model(self, model_path: str, model_name: str):
        super().stage_model(model_path, model_name)
