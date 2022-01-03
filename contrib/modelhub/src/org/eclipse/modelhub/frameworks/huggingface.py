import os
from org.eclipse.modelhub.model_hub import model_hub_dir, ModelHub
from huggingface_hub import hf_hub_url,cached_download

framework_name = 'huggingface'
framework_dir = os.path.join(model_hub_dir, framework_name)
BASE_URL = 'https://huggingface.co'

class HuggingFaceModelHub(ModelHub):
    def __init__(self):
        super().__init__(framework_name,BASE_URL)

    def download_model(self, model_path):
        url = hf_hub_url(repo_id=model_path, filename="config.json")
        cached_download(url)

    def stage_model(self, model_path: str, model_name: str):
        super().stage_model(model_path, model_name)