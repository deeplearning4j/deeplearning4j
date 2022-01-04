import os

from tensorflow.core.framework.graph_pb2 import GraphDef

from omnihub.model_hub import model_hub_dir, ModelHub
import tarfile
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow as tf
import tempfile

framework_name = 'tensorflow'
framework_dir = os.path.join(model_hub_dir, framework_name)
BASE_URL = 'https://tfhub.dev'


def convert_saved_model(saved_model_dir) -> GraphDef:
    """
    Convert the saved model (expanded as a directory)
    to a frozen graph def
    :param saved_model_dir: the input model directory
    :return:  the loaded graph def with all parameters in the model
    """
    saved_model = tf.saved_model.load(saved_model_dir)
    graph_def = saved_model.signatures['serving_default']
    frozen = convert_variables_to_constants_v2(graph_def)
    return frozen.graph.as_graph_def()


class TensorflowModelHub(ModelHub):
    def __init__(self):
        super().__init__(framework_name, BASE_URL)

    def download_model(self, model_path, **kwargs):
        final_name = model_path.split('/')[-2]
        model_path = super().download_model(model_path + '?tf-hub-format=compressed')
        if not tarfile.is_tarfile(model_path):
            raise Exception(f'Unable to open tar file at path {model_path}')

        mode = kwargs.get('mode', 'r:gz')
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(model_path, mode=mode) as downloaded:
                downloaded.extractall(tmpdir)
                tf.io.write_graph(convert_saved_model(tmpdir), framework_dir, f'{final_name}.pb', as_text=False)
