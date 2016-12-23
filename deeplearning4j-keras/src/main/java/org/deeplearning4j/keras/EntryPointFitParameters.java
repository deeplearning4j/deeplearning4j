package org.deeplearning4j.keras;

import lombok.Builder;

@Builder
public class EntryPointFitParameters {
    private String modelFilePath;
    private String type;
    private String trainFeaturesFile;
    private String trainLabelsFile;
    private int batchSize;
    private long nbEpoch;
    private String validationXFilePath;
    private String validationYFilePath;
    private String dimOrdering;

    public String getModelFilePath() {
        return modelFilePath;
    }

    public String getType() {
        return type;
    }

    public String getTrainFeaturesFile() {
        return trainFeaturesFile;
    }

    public String getTrainLabelsFile() {
        return trainLabelsFile;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public long getNbEpoch() {
        return nbEpoch;
    }

    public String getValidationXFilePath() {
        return validationXFilePath;
    }

    public String getValidationYFilePath() {
        return validationYFilePath;
    }

    public String getDimOrdering() {
        return dimOrdering;
    }
}
