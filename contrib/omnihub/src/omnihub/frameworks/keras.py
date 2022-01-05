import os

from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.applications.mobilenet_v3 import MobileNetV3

from omnihub.model_hub import omnihub_dir, ModelHub
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet50, ResNet101, ResNet152
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, \
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.nasnet import NASNetLarge, NASNetMobile
from tensorflow.keras.applications.xception import Xception
framework_name = 'keras'
framework_dir = os.path.join(omnihub_dir, framework_name)
BASE_URL = 'https://storage.googleapis.com/tensorflow/keras-applications'
keras_path = os.path.join(os.path.expanduser('~'), '.keras','models')


class KerasModelHub(ModelHub):
    def __init__(self):
        super().__init__(framework_name, BASE_URL)

    def download_model(self, model_path, **kwargs) -> str:
        """
        Download the model and return the model path on the file system
        :param model_path:  the model path for the URL
        :param kwargs: various kwargs for customizing the underlying behavior
        :return:  the local file path
        """
        model_path = self.download_for_url(model_path,**kwargs)
        return model_path

    def stage_model(self, model_path: str, model_name: str):
        super().stage_model(model_path, model_name)

    def download_for_url(self, path: str,**kwargs):
        """
        Download the file at the given URL
        :param path:  the path to download
        :param kwargs:  various kwargs for customizing the underlying behavior of
        the model download and setup
        :return: the absolute path to the model
        """
        path_split = path.split('/')
        type = path_split[0]
        weights_file = path_split[1]
        include_top = 'no_top' in weights_file
        if type == 'vgg19':
            ret = VGG19(include_top=include_top, **kwargs)
        elif type == 'vgg16':
            ret = VGG16(include_top=include_top, **kwargs)
        elif type == 'resnet50':
            ret = ResNet50(include_top=include_top, **kwargs)
        elif type == 'resnet101':
            ret = ResNet101(include_top=include_top, **kwargs)
        elif type == 'resnet152':
            ret = ResNet152(include_top=include_top, **kwargs)
        elif type == 'resnet50v2':
            ret = ResNet50V2(include_top=include_top, **kwargs)
        elif type == 'resnet101v2':
            ret = ResNet101V2(include_top=include_top, **kwargs)
        elif type == 'resnet152v2':
            ret = ResNet152V2(include_top=include_top, **kwargs)
        elif type == 'densenet121':
            ret = DenseNet121(include_top=include_top)
        elif type == 'densenet169':
            ret = DenseNet169(include_top=include_top, **kwargs)
        elif type == 'densenet201':
            ret = DenseNet201(include_top=include_top, **kwargs)
        elif type == 'inceptionresnetv2':
            ret = InceptionResNetV2(include_top=include_top, **kwargs)
        elif type == 'efficientnetb0':
            ret = EfficientNetB0(include_top=include_top, **kwargs)
        elif type == 'efficientnetb1':
            ret = EfficientNetB1(include_top=include_top, **kwargs)
        elif type == 'efficientnetb2':
            ret = EfficientNetB2(include_top=include_top, **kwargs)
        elif type == 'efficientnetb3':
            ret = EfficientNetB3(include_top=include_top, **kwargs)
        elif type == 'efficientnetb4':
            ret = EfficientNetB4(include_top=include_top, **kwargs)
        elif type == 'efficientnetb5':
            ret = EfficientNetB5(include_top=include_top, **kwargs)
        elif type == 'efficientnetb6':
            ret = EfficientNetB6(include_top=include_top, **kwargs)
        elif type == 'efficientnetb8':
            efficient_net = EfficientNetB7(include_top=include_top, **kwargs)
        elif type == 'mobilenet':
            ret = MobileNet(include_top=include_top, **kwargs)
        elif type == 'mobilenetv2':
            ret = MobileNetV2(include_top=include_top)
        elif type == 'mobilenetv3':
            mobile_net = MobileNetV3(include_top=include_top, **kwargs)
        elif type == 'inceptionv3':
            ret = InceptionV3(include_top=include_top, **kwargs)
        elif type == 'nasnet':
            ret = NASNetLarge(include_top=include_top, **kwargs)
        elif type == 'nasnet_mobile':
            ret = NASNetMobile(include_top=include_top, **kwargs)
        elif type == 'xception':
            ret = Xception(include_top=include_top, **kwargs)
        model_path = os.path.join(keras_path, weights_file)
        return model_path
