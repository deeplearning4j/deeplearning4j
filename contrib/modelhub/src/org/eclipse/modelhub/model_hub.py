import os
import shutil
from typing import IO

import requests

MODEL_HUB_DIR = 'MODEL_HUB_DIR'
if os.environ.__contains__(MODEL_HUB_DIR):
    model_hub_dir = os.environ[MODEL_HUB_DIR]
else:
    model_hub_dir = os.path.join(os.path.expanduser('~'), '.model_hub')

if not os.path.exists(model_hub_dir):
    os.mkdir(model_hub_dir)


class ModelHub(object):
    def __init__(self, framework_name: str, base_url: str):
        self.framework_name = framework_name
        self.stage_model_dir = os.path.join(model_hub_dir, self.framework_name)
        if not os.path.exists(self.stage_model_dir):
            os.mkdir(self.stage_model_dir)
        self.base_url = base_url

    def _download_file(self, url: str):
        local_filename = os.path.join(self.stage_model_dir, url.split('/')[-1])
        # NOTE the stream=True parameter below
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    # if chunk:
                    f.write(chunk)
        return local_filename

    def download_model(self, model_path) -> str:
        """
        Meant to be overridden by sub classes.
        Handles downloading a model with the target URL
        at the path specified.
        :param model_path:  the path to the model from the base URL of the web service
        :return: the path to the original model
        """
        model_path = self._download_file(f'{self.base_url}/{model_path}')
        return model_path

    def stage_model(self, model_path: str, model_name: str):
        """
        Copy the model from its original path to the target
        directory under self.stage_model_dir
        :param model_path: the original path to the model downloaded
        by the underlying framework
        :param model_name: the name of the model file to save as
        :return:
        """
        shutil.copy(model_path, os.path.join(self.stage_model_dir, model_name))

    def stage_model_stream(self, model_path: IO, model_name: str):
        """
        Copy the model from its original path to the target
        directory under self.stage_model_dir
        :param model_path: the original path to the model downloaded
        by the underlying framework
        :param model_name: the name of the model file to save as
        :return:
        """
        with open(os.path.join(self.stage_model_dir, model_name), 'wb+') as f:
            shutil.copyfileobj(model_path, f)
