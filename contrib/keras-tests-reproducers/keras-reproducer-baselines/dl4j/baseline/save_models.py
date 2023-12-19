import argparse
import os
import sys
import pkgutil
from model_manager import ModelManager
from typing import Type
from tests.bidirectional_test import BidirectionalModelManager
managers = [
    BidirectionalModelManager()
]
def main():
    parser = argparse.ArgumentParser(description='Save ModelManager subclasses.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to store models')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    baseline_module = __import__('tests')  # Import dl4j/baseline
    baseline_path = os.path.dirname(baseline_module.__file__)  # Get its path

      # Instantiate the ModelManager subclass, and build the models
    for model_manager in managers:
        model_manager.model_builder()
        model_manager.set_default_inputs()
        model_manager.compute_all_outputs()
        model_manager.compute_all_gradients()
        model_manager.save_all(args.output_dir)

    # Save all the models
        for model_name, model_state in model_manager.models.items():
            save_dir = os.path.join(args.output_dir, model_name)
            os.makedirs(save_dir, exist_ok=True)
            model_state.model.save(os.path.join(save_dir, f"{model_name}.h5"))

if __name__ == "__main__":
    main()