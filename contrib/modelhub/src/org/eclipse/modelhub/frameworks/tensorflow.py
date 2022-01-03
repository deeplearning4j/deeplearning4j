import os
from org.eclipse.modelhub.model_hub import model_hub_dir, ModelHub
import tarfile

framework_name = 'tensorflow'
framework_dir = os.path.join(model_hub_dir, framework_name)
BASE_URL = 'https://tfhub.dev'


class TensorflowModelHub(ModelHub):
    def __init__(self):
        super().__init__(framework_name, BASE_URL)

    def download_model(self, model_path):
        final_name = model_path.split('/')[-2]
        model_path = super().download_model(model_path + '?tf-hub-format=compressed')
        if not tarfile.is_tarfile(model_path):
            raise Exception(f'Unable to open tar file at path {model_path}')
        mode = 'r'
        if model_path.endswith('.gz'):
            mode = 'r:gz'

        with tarfile.open(model_path, mode=mode) as downloaded:
            for member in downloaded:
                if '.pb' in member.name:
                    with downloaded.extractfile(member) as extracted_member:
                        input_name = member.name
                        if input_name.startswith('./'):
                            input_name = input_name[2:]
                        elif input_name.startswith('/'):
                            input_name = input_name[1:]
                        super().stage_model_stream(extracted_member, f'{final_name}_{input_name}')


