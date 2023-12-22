import argparse
import os

import keras.utils
from tests.bidirectional_test import BidirectionalModelManager
import tempfile
managers = [
    BidirectionalModelManager()
]


def main():
    # ensure random weights are reproducible
    keras.utils.set_random_seed(42)
    parser = argparse.ArgumentParser(description='Save ModelManager subclasses.')
    parser.add_argument('--output_dir', type=str, required=False,
                        help='Directory to store models',default=os.path.join(tempfile.gettempdir(),'keras-dl4j-verification-models'))
    args = parser.parse_args()



    os.makedirs(args.output_dir, exist_ok=True)

    # Instantiate the ModelManager subclass, and build the models
    for model_manager in managers:
        # make the directory for the test in the output models are stored here from the root
        model_manager.make_dir(args.output_dir)
        # build the models
        model_manager.model_builder()
        # set the default inputs
        model_manager.set_default_inputs()
        model_manager.compute_all_outputs()
        # save to the parent directory passed in earlier with a subdirectory of the test name.
        model_manager.save_all()


if __name__ == "__main__":
    main()
