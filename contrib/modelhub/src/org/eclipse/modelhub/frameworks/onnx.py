import os
from org.eclipse.modelhub.model_hub import model_hub_dir, ModelHub

framework_name = 'onnx'
framework_dir = os.path.join(model_hub_dir, framework_name)
BASE_URL = 'https://media.githubusercontent.com/media/onnx/models/master'


class OnnxModelHub(ModelHub):
    def __init__(self):
        super().__init__(framework_name, BASE_URL)
