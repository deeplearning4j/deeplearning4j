import os
from org.eclipse.modelhub.model_hub import model_hub_dir, ModelHub
from model_archiver.manifest_components import manifest,model
import torch
from  torchvision import models
from torchvision.models.detection import FasterRCNN
import numpy as np

framework_name = 'pytorch'
framework_dir = os.path.join(model_hub_dir, framework_name)
BASE_URL = 'https://s3.amazonaws.com/pytorch/models'


class PytorchModelHub(ModelHub):
    def __init__(self):
        super().__init__(framework_name,BASE_URL)


    def download_model(self, model_path,**kwargs) -> str:
        model = None
        if model_path == 'resnet18':
            height = kwargs.get('height',224)
            width = kwargs.get('width',224)
            x = torch.from_numpy(np.ones((1, 3, height, width),dtype=np.float32))
            model = models.resnet18(pretrained=True,**kwargs)
        elif model_path == 'alexnet':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.alexnet(pretrained=True,**kwargs)
        elif model_path == 'vgg16':
            height = kwargs.get('height', 224)
            width = kwargs.get('width', 224)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.vgg16(pretrained=True,**kwargs)
        elif model_path == 'squeezenet1_0':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.squeezenet1_0(pretrained=True,**kwargs)
        elif model_path == 'densenet161':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.densenet161()
        elif model_path == 'inception_v3':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.inception_v3(pretrained=True,**kwargs)
        elif model_path == 'googlenet':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.googlenet(pretrained=True,**kwargs)
        elif model_path == 'shufflenet_v2_x1_0':
            height = kwargs.get('height', 224)
            width = kwargs.get('width', 224)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.shufflenet_v2_x1_0(pretrained=True,**kwargs)
        elif model_path == 'mobilenet_v2':
            height = kwargs.get('height', 32)
            width = kwargs.get('width', 32)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.mobilenet_v2(pretrained=True,**kwargs)
        elif model_path == 'mobilenet_v3_large':
            height = kwargs.get('height', 320)
            width = kwargs.get('width', 320)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.mobilenet_v3_large(pretrained=True,**kwargs)
        elif model_path == 'mobilenet_v3_small':
            height = kwargs.get('height', 320)
            width = kwargs.get('width', 320)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.mobilenet_v3_small(pretrained=True,**kwargs)
        elif model_path == 'resnext50_32x4d':
            height = kwargs.get('height', 224)
            width = kwargs.get('width', 224)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.resnext50_32x4d(pretrained=True,**kwargs)
        elif model_path == 'wide_resnet50_2':
            height = kwargs.get('height', 224)
            width = kwargs.get('width', 224)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.wide_resnet50_2(pretrained=True,**kwargs)
        elif model_path == 'mnasnet1_0':
            height = kwargs.get('height', 224)
            width = kwargs.get('width', 224)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.mnasnet1_0(pretrained=True,**kwargs)
        elif model_path == 'efficientnet_b0':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.efficientnet_b0(pretrained=True,**kwargs)
        elif model_path == 'efficientnet_b1':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.efficientnet_b1(pretrained=True,**kwargs)
        elif model_path == 'efficientnet_b2':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.efficientnet_b2(pretrained=True,**kwargs)
        elif model_path == 'efficientnet_b3':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.efficientnet_b3(pretrained=True,**kwargs)
        elif model_path == 'efficientnet_b4':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.efficientnet_b4(pretrained=True,**kwargs)
        elif model_path == 'efficientnet_b5':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.efficientnet_b5(pretrained=True,**kwargs)
        elif model_path == 'efficientnet_b6':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.efficientnet_b6(pretrained=True,**kwargs)
        elif model_path == 'efficientnet_b7':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.efficientnet_b7(pretrained=True,**kwargs)
        elif model_path == 'regnet_y_400mf':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.regnet_y_400mf(pretrained=True,**kwargs)
        elif model_path == 'regnet_y_800mf':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.regnet_y_800mf(pretrained=True,**kwargs)
        elif model_path == 'regnet_y_1_6gf':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.regnet_y_1_6gf(pretrained=True,**kwargs)
        elif model_path == 'regnet_y_3_2gf':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.regnet_y_3_2gf(pretrained=True,**kwargs)
        elif model_path == 'regnet_y_8gf':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.regnet_y_8gf(pretrained=True,**kwargs)
        elif model_path == 'regnet_y_16gf':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.regnet_y_16gf(pretrained=True,**kwargs)
        elif model_path == 'regnet_y_32gf':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.regnet_y_32gf(pretrained=True,**kwargs)
        elif model_path == 'regnet_x_400mf':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.regnet_x_400mf(pretrained=True,**kwargs)
        elif model_path == 'regnet_x_800mf':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.regnet_x_800mf(pretrained=True,**kwargs)
        elif model_path == 'regnet_x_1_6gf':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.regnet_x_1_6gf(pretrained=True,**kwargs)
        elif model_path == 'regnet_x_3_2gf':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.regnet_x_3_2gf(pretrained=True,**kwargs)
        elif model_path == 'regnet_x_8gf':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.regnet_x_8gf(pretrained=True,**kwargs)
        elif model_path == 'regnet_x_16gf':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.regnet_x_16gf(pretrained=True,**kwargs)
        elif model_path == 'regnet_x_32gf':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.regnet_x_32gf(pretrained=True,**kwargs)
        elif model_path == 'fasterrcnn':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.detection.FasterRCNN(pretrained=True,**kwargs)
        elif model_path == 'ssd':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.detection.SSD(pretrained=True,**kwargs)
        elif model_path == 'retinanet':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.detection.RetinaNet(pretrained=True,**kwargs)
        elif model_path == 'maskrcnn':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.detection.MaskRCNN(pretrained=True,**kwargs)
        elif model_path == 'keypointrcnn':
            height = kwargs.get('height', 256)
            width = kwargs.get('width', 256)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.detection.KeypointRCNN(pretrained=True,**kwargs)
        elif model_path == 'retinanet':
            height = kwargs.get('height', 512)
            width = kwargs.get('width', 512)
            x = torch.from_numpy(np.ones((1,3,height,width),dtype=np.float32))
            model = models.detection.RetinaNet(pretrained=True,**kwargs)
        print('done downloading')
        torch.onnx.export(model,
                          x,
                          f'{framework_dir}/{model_path}.onnx',
                          export_params=True,
                          opset_version=13,
                          **kwargs)

    def stage_model(self, model_path: str, model_name: str):
        super().stage_model(model_path, model_name)