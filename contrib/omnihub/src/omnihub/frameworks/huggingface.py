import os

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from omnihub.model_hub import omnihub_dir, ModelHub
framework_name = 'huggingface'
framework_dir = os.path.join(omnihub_dir, framework_name)
BASE_URL = 'https://huggingface.co'
from transformers import AutoTokenizer, AutoModel, TFAutoModel
import tensorflow as tf
import torch

class HuggingFaceModelHub(ModelHub):
    def __init__(self):
        """
        Note that when downloading models for usage from huggingface the URLs should take a very specific format.
        Since huggingface spaces uses git LFS it uses branch names. Huggingface spaces defaults to the main branch.
        Typically, the URL formula is: https://huggingface.co + the repo name followed by resolve/main/file_name
        This file name should be a path to a raw model file. For specific framework tools, feel free to reuse
        code in this repository
        """
        super().__init__(framework_name, BASE_URL)

    def resolve_url(self, repo_path, file_name, branch_name='main'):
        """
        Resolve the file name for downloading from huggingface hub.
        This creates a path using a branch name with a default of main
        for downloading models from the hub.
        :param repo_path: repo path to download from. This is usually
        the namespace after  huggingface.co
        :param file_name:  the file name to download, this should be a model file
        :param branch_name: the branch name (defaults to main)
        :return:  the real url to use for downloading the target file
        """
        return f'{repo_path}/resolve/{branch_name}/{file_name}'

    def _download_tf(self,model_path):
        output_model = TFAutoModel.from_pretrained(model_path)
        return output_model
    def _download_pytorch(self,model_path):
        output_model = AutoModel.from_pretrained(model_path)
        dummy_inputs = output_model.dummy_inputs
        inputs_ordered = []
        non_main_inputs = []
        for name, array in dummy_inputs.items():
            if name == output_model.main_input_name:
                inputs_ordered.append(array)
            else:
                non_main_inputs.append(array)
            ordred_dummy_inputs = inputs_ordered + non_main_inputs
        return output_model, tuple(ordred_dummy_inputs)

    def download_model(self, model_path, **kwargs) -> str:
        """
        Download the model for the given path.
        A framework_name kwarg is required in order to
        put the model in the proper location.
        Due to the nature of huggingface repos being multi framework
        it's up to the user to specify where a file should go.
        Valid frameworks are:
        onnx
        pytorch
        tensorflow
        keras
        :param model_path: the path to the model to download
        :param kwargs:  a kwargs containing framework_name as described above
        :return:  the path to the model
        """
        assert 'framework_name' in kwargs
        model_name = model_path.split('/')[-1]
        framework_name = kwargs.pop('framework_name')

        if framework_name == 'keras' or framework_name == 'tensorflow':
            output_model = TFAutoModel.from_pretrained(model_path)
            callable = tf.function(output_model.call)
            concrete_function = callable.get_concrete_function(output_model.dummy_inputs)
            frozen = convert_variables_to_constants_v2(concrete_function)
            graph_def = frozen.graph.as_graph_def()
            tf.io.write_graph(graph_def, os.path.join(omnihub_dir,'tensorflow'), f'{model_name}.pb', as_text=False)
        else:
            download_function = kwargs.get('download_function', self._download_pytorch)
            if 'download_function' in kwargs:
                kwargs.pop('download_function')
            output_model,dummy_inputs = download_function(model_path)
            torch.onnx.export(output_model,
                              dummy_inputs,
                              f'{os.path.join(omnihub_dir,framework_name)}/{model_name}.onnx',
                              export_params=True,
                              do_constant_folding=False,
                              opset_version=13,
                              **kwargs)



    def stage_model(self, model_path: str, model_name: str):
        super().stage_model(model_path, model_name)
