import os
from omnihub.model_hub import omnihub_dir, ModelHub

framework_name = 'onnx'
framework_dir = os.path.join(omnihub_dir, framework_name)
BASE_URL = 'https://media.githubusercontent.com/media/onnx/models/master'


class OnnxModelHub(ModelHub):
    def __init__(self):
        super().__init__(framework_name, BASE_URL)
