import os

from tensorflow.python.framework.graph_util_impl import convert_variables_to_constants

from org.eclipse.modelhub.model_hub import model_hub_dir, ModelHub
import tarfile
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow as tf
import tempfile
import chardet
framework_name = 'tensorflow'
framework_dir = os.path.join(model_hub_dir, framework_name)
BASE_URL = 'https://tfhub.dev'

import codecs
utf8reader = codecs.getreader('utf-8')
def force_decode(string, codecs=['utf8', 'cp1252']):
    for i in codecs:
        try:
            return string.decode(i)
        except UnicodeDecodeError:
            pass


import cchardet
def convert_encoding(data, new_coding = 'UTF-8'):
  encoding = cchardet.detect(data)['encoding']

  if new_coding.upper() != encoding.upper():
    data = data.decode(encoding, data).encode(new_coding)

  return data

class TensorflowModelHub(ModelHub):
    def __init__(self):
        super().__init__(framework_name, BASE_URL)

    def download_model(self, model_path,**kwargs):
        final_name = model_path.split('/')[-2]
        model_path = super().download_model(model_path + '?tf-hub-format=compressed')
        if not tarfile.is_tarfile(model_path):
            raise Exception(f'Unable to open tar file at path {model_path}')

        mode = kwargs.get('mode','r:gz')
        with  tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(model_path, mode=mode) as downloaded:
                downloaded.extractall(tmpdir)
                # See: https://ersanpreet.wordpress.com/2019/06/25/converting-tensorflow-saved-model-to-model-graphdef/
                saved_model = tf.saved_model.load(tmpdir)
                graph_def = saved_model.signatures['serving_default']
                frozen = convert_variables_to_constants_v2(graph_def)
                tf.io.write_graph(frozen.graph.as_graph_def(), framework_dir, f'{final_name}.pb', as_text=False)





