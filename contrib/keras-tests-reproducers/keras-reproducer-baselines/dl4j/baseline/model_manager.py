import abc
import os
from typing import Dict, List
import tensorflow as tf
import numpy as np


class ModelState:
    def __init__(self, model_name: str, model: tf.keras.models.Model, inputs: List[tf.Tensor] = [],
                 outputs: List[tf.Tensor] = [], gradients: List[List[tf.Tensor]] = []):
        self.model_name = model_name
        self.model = model
        self.inputs = inputs
        self.outputs = outputs
        self.gradients = gradients


class ModelManager(abc.ABC):
    def __init__(self):
        self.models: Dict[str, ModelState] = {}

    def add_model(self, model_name: str, model: tf.keras.models.Model) -> None:
        self.models[model_name] = ModelState(model_name, model)

    @abc.abstractmethod
    def model_builder(self):
        pass

    @abc.abstractmethod
    def set_default_inputs(self) -> None:
        pass


    def compute_all_gradients(self) -> None:
        for model_name in self.models.keys():
            self.compute_gradients(model_name)

    def compute_all_outputs(self) -> None:
        for model_name in self.models.keys():
            self.compute_outputs(model_name)
    def set_inputs(self, model_name: str, inputs: List[tf.Tensor]) -> None:
        self.models[model_name].inputs = inputs

    def compute_outputs(self, model_name: str) -> None:
        model_state = self.models.get(model_name)
        if not model_state:
            raise ValueError(f"Model not found for model: {model_name}")

        model_state.outputs = model_state.model.predict(model_state.inputs)

    def compute_gradients(self, model_name: str) -> None:
        model_state = self.models.get(model_name)
        model_state.gradients = []
        inputs = model_state.inputs
        if inputs is not None:
            for input_ in inputs:
                with tf.GradientTape() as tape:
                    tape.watch(model_state.model.trainable_variables)
                    predictions = model_state.model(input_, training=True)
                    loss = model_state.model.loss(tf.zeros_like(predictions), predictions)
                grads = tape.gradient(loss, model_state.model.trainable_variables)
                model_state.gradients.append(grads)

    def save_all(self, output_dir: str) -> None:
        for model_name, model_state in self.models.items():
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Save the model in .h5 format
            model_state.model.save(os.path.join(model_dir, "model.h5"), save_format='h5')

            # Save the inputs, outputs, and gradients
            inputs, outputs, gradients = model_state.inputs, model_state.outputs, model_state.gradients
            for i, (input_, output, gradients_) in enumerate(zip(inputs, outputs, gradients)):
                np.save(os.path.join(model_dir, f"{model_name}_input_{i}.npy"), input_.numpy())
                np.save(os.path.join(model_dir, f"{model_name}_output_{i}.npy"), output)
                for j, grad in enumerate(gradients_):
                    np.save(os.path.join(model_dir, f"{model_name}_gradient_{j}.npy"), grad.numpy())
