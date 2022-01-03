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

    def download_model(self, model_path):
        ##super_path = super().download_model(model_path)
        model = None
        if model_path == 'resnet18':
            x = torch.from_numpy(np.ones((1, 3, 224, 224)))
            model = models.resnet18(pretrained=True)
        elif model_path == 'alexnet':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.alexnet(pretrained=True)
        elif model_path == 'vgg16':
            x = torch.from_numpy(np.ones((1,3,224,224)))
            model = models.vgg16(pretrained=True)
        elif model_path == 'squeezenet1_0':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.squeezenet1_0(pretrained=True)
        elif model_path == 'densenet161':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.densenet161()
        elif model_path == 'inception_v3':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.inception_v3(pretrained=True)
        elif model_path == 'googlenet':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.googlenet(pretrained=True)
        elif model_path == 'shufflenet_v2_x1_0':
            x = torch.from_numpy(np.ones((1,3,224,224)))
            model = models.shufflenet_v2_x1_0(pretrained=True)
        elif model_path == 'mobilenet_v2':
            x = torch.from_numpy(np.ones((1,3,32,32)))
            model = models.mobilenet_v2(pretrained=True)
        elif model_path == 'mobilenet_v3_large':
            x = torch.from_numpy(np.ones((1,3,320,320)))
            model = models.mobilenet_v3_large(pretrained=True)
        elif model_path == 'mobilenet_v3_small':
            x = torch.from_numpy(np.ones((1,3,320,320)))
            model = models.mobilenet_v3_small(pretrained=True)
        elif model_path == 'resnext50_32x4d':
            x = torch.from_numpy(np.ones((1,3,224,224)))
            model = models.resnext50_32x4d(pretrained=True)
        elif model_path == 'wide_resnet50_2':
            x = torch.from_numpy(np.ones((1,3,224,224)))
            model = models.wide_resnet50_2(pretrained=True)
        elif model_path == 'mnasnet1_0':
            x = torch.from_numpy(np.ones((1,3,224,224)))
            model = models.mnasnet1_0(pretrained=True)
        elif model_path == 'efficientnet_b0':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.efficientnet_b0(pretrained=True)
        elif model_path == 'efficientnet_b1':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.efficientnet_b1(pretrained=True)
        elif model_path == 'efficientnet_b2':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.efficientnet_b2(pretrained=True)
        elif model_path == 'efficientnet_b3':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.efficientnet_b3(pretrained=True)
        elif model_path == 'efficientnet_b4':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.efficientnet_b4(pretrained=True)
        elif model_path == 'efficientnet_b5':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.efficientnet_b5(pretrained=True)
        elif model_path == 'efficientnet_b6':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.efficientnet_b6(pretrained=True)
        elif model_path == 'efficientnet_b7':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.efficientnet_b7(pretrained=True)
        elif model_path == 'regnet_y_400mf':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.regnet_y_400mf(pretrained=True)
        elif model_path == 'regnet_y_800mf':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.regnet_y_800mf(pretrained=True)
        elif model_path == 'regnet_y_1_6gf':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.regnet_y_1_6gf(pretrained=True)
        elif model_path == 'regnet_y_3_2gf':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.regnet_y_3_2gf(pretrained=True)
        elif model_path == 'regnet_y_8gf':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.regnet_y_8gf(pretrained=True)
        elif model_path == 'regnet_y_16gf':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.regnet_y_16gf(pretrained=True)
        elif model_path == 'regnet_y_32gf':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.regnet_y_32gf(pretrained=True)
        elif model_path == 'regnet_x_400mf':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.regnet_x_400mf(pretrained=True)
        elif model_path == 'regnet_x_800mf':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.regnet_x_800mf(pretrained=True)
        elif model_path == 'regnet_x_1_6gf':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.regnet_x_1_6gf(pretrained=True)
        elif model_path == 'regnet_x_3_2gf':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.regnet_x_3_2gf(pretrained=True)
        elif model_path == 'regnet_x_8gf':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.regnet_x_8gf(pretrained=True)
        elif model_path == 'regnet_x_16gf':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.regnet_x_16gf(pretrained=True)
        elif model_path == 'regnet_x_32gf':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.regnet_x_32gf(pretrained=True)
        elif model_path == 'fasterrcnn':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.detection.FasterRCNN(pretrained=True)
        elif model_path == 'ssd':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.detection.SSD(pretrained=True)
        elif model_path == 'retinanet':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.detection.RetinaNet(pretrained=True)
        elif model_path == 'maskrcnn':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.detection.MaskRCNN(pretrained=True)
        elif model_path == 'keypointrcnn':
            x = torch.from_numpy(np.ones((1,3,256,256)))
            model = models.detection.KeypointRCNN(pretrained=True)
        elif model_path == 'retinanet':
            x = torch.from_numpy(np.ones((1,3,512,512)))
            model = models.detection.RetinaNet(pretrained=True)
        print('done downloading')
       # torch.onnx.export(model,x,f'{model_path}.onnx',export_params=True,opset_version=13,input_names=[],output_names=[],  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
       #                         'output' : {0 : 'batch_size'}})

    def stage_model(self, model_path: str, model_name: str):
        super().stage_model(model_path, model_name)