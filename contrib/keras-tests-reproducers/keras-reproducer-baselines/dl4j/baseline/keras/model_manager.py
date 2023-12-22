import abc
import os
from typing import Dict, List
import tensorflow as tf
import numpy as np
from keras import Sequential, Model
from keras.engine.functional import Functional

"""
The state of a model, including the model itself, the inputs, and the outputs
"""


class ModelState:
    def __init__(self, model_name: str, model: tf.keras.models.Model, inputs: List[tf.Tensor] = [],
                 outputs: List[tf.Tensor] = []):
        self.model_name = model_name
        self.model = model
        self.inputs = inputs
        self.outputs = outputs


"""

The ModelManager class is responsible for building and saving models. It is an abstract class, and subclasses
must implement the model_builder and set_default_inputs methods. The model_builder method is responsible for
building the models and adding them to the model manager. The set_default_inputs method is responsible for
setting the default inputs for the models. The compute_all_outputs method is responsible for computing the outputs
for all models. The save_all method is responsible for saving all models and inputs/outputs to the output directory.

"""


class ModelManager(abc.ABC):
    def __init__(self):
        self.models: Dict[str, ModelState] = {}
        self.base_dir = ''

    def make_dir(self, path: str) -> None:
        os.makedirs(os.path.join(path, self.test_name()), exist_ok=True)
        self.base_dir = os.path.join(path, self.test_name())

    def test_name(self) -> str:
        """
        :return: the name of the test - this is typically used as the directory.
        :return: the class name
        """
        return self.__class__.__name__

    def add_model(self, model_name: str, model: tf.keras.models.Model) -> None:
        """
        Adds a model to the model manager
        :param model_name: the name of the model
        :param model: the model itself
        :return: none
        """
        self.models[model_name] = ModelState(model_name, model)

    @abc.abstractmethod
    def model_builder(self):
        """
        Builds the models and adds them to the model manager
        :return:
        """
        pass

    @abc.abstractmethod
    def set_default_inputs(self) -> None:
        """
        Sets the default inputs for the models
        :return:
        """
        pass

    def compute_all_outputs(self) -> None:
        """
        Computes the outputs for all models
        :return:
        """
        for model_name in self.models.keys():
            self.compute_outputs(model_name)

    def set_inputs(self, model_name: str, inputs: List[tf.Tensor]) -> None:
        self.models[model_name].inputs = inputs

    def compute_outputs(self, model_name: str) -> None:
        """
        Computes the outputs for a single model
        :param model_name:
        :return:
        """
        model_state = self.models.get(model_name)
        if not model_state:
            raise ValueError(f"Model not found for model: {model_name}")

        model_state.outputs = model_state.model(model_state.inputs)

    def save_all(self) -> None:
        """
        Saves all models and inputs/outputs to the output directory
        :return:
        """
        for model_name, model_state in self.models.items():
            model_dir = os.path.join(self.base_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Save the model in .h5 format
            model_state.model.save(os.path.join(model_dir, "model.h5"), save_format='h5')
            if not isinstance(model_state.inputs, list):
                model_state.inputs = [model_state.inputs]
            if not isinstance(model_state.outputs, list):
                model_state.outputs = [model_state.outputs]

            # Save the inputs, outputs, and gradients
            inputs, outputs = model_state.inputs, model_state.outputs
            for i, (input_, output) in enumerate(zip(inputs, outputs)):
                np.save(os.path.join(model_dir, f"{model_name}_input_{i}.npy"), input_.numpy())
                np.save(os.path.join(model_dir, f"{model_name}_output_{i}.npy"), output.numpy())

            # note this is fairly simplistic, all we want to do here
            # is have a human readable way of knowing what the model is without
            # a bunch of json and hdf5 complexity
            if type(model_state.model) is Sequential:
                with open(os.path.join(model_dir, "model_type.txt"), 'w') as f:
                    f.write('Sequential')
            elif type(model_state.model) is Functional:
                with open(os.path.join(model_dir, "model_type.txt"), 'w') as f:
                    f.write('Functional')
            elif type(model_state.model) is Model:
                with open(os.path.join(model_dir, "model_type.txt"), 'w') as f:
                    f.write('Model')
