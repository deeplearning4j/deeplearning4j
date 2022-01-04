import os
from org.eclipse.modelhub.model_hub import model_hub_dir, ModelHub
from huggingface_hub import hf_hub_url,cached_download

framework_name = 'huggingface'
framework_dir = os.path.join(model_hub_dir, framework_name)
BASE_URL = 'https://huggingface.co'

class HuggingFaceModelHub(ModelHub):
    def __init__(self):
        """
        Note that when downloading models for usage from huggingface the URLs should take a very specific format.
        Since huggingface spaces uses git LFS it uses branch names. Huggingface spaces defaults to the main branch.
        Typically the URL formula is: https://huggingface.co + the repo name followed by resolve/main/file_name
        This file name should be a path to a raw model file. For specific framework tools, feel free to reuse
        code in this repository
        """
        super().__init__(framework_name,BASE_URL)

    def download_model(self, model_path,**kwargs) -> str:
        return super().download_model(model_path)

    def stage_model(self, model_path: str, model_name: str):
        super().stage_model(model_path, model_name)