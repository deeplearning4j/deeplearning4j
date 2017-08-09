package org.deeplearning4j.nn.modelimport.keras.config;

import lombok.Data;

@Data
public class KerasModelConfiguration {

    /* Model meta information fields */
    private final String fieldClassName = "class_name";
    private final String fieldClassNameSequential = "Sequential";
    private final String fieldClassNameModel = "Model";
    private final String fieldKerasVersion = "keras_version";

    /* Model configuration field. */
    private final String modelFieldConfig = "config";
    private final String modelFieldLayers = "layers";
    private final String modelFieldInputLayers = "input_layers";
    private final String modelFieldOutputLayers = "output_layers";

    /* Training configuration field. */
    private final String trainingLoss = "loss";
    private final String trainingWeightsRoot = "model_weights";
    private final String trainingModelConfigAttribute = "model_config";
    private final String trainingTrainingConfigAttribute = "training_config";

}
