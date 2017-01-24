package org.deeplearning4j.keras.model;

import org.deeplearning4j.keras.model.KerasModelType;

public class KerasModelRef {

    protected String modelFilePath;
    protected KerasModelType modelType;

    public KerasModelRef(String modelFilePath, KerasModelType modelType) {
        this.modelFilePath = modelFilePath;
        this.modelType = modelType;
    }

    public String getModelPath() { return modelFilePath; }

    public KerasModelType getModelType() { return modelType; }

}
