import os
from org.eclipse.modelhub.model_hub import model_hub_dir, ModelHub

framework_name = 'huggingface'
framework_dir = os.path.join(model_hub_dir, framework_name)
BASE_URL = 'https://huggingface.co'


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

    def resolve_url(self,repo_path,file_name,branch_name='main'):
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

    def download_model(self, model_path, **kwargs) -> str:
        return super().download_model(model_path)

    def stage_model(self, model_path: str, model_name: str):
        super().stage_model(model_path, model_name)
