package org.deeplearning4j.keras;

import lombok.Builder;
import lombok.Data;

/**
 * POJO with parameters of the `fit` method of available through the py4j Python-Java bridge
 */
@Data
@Builder
public class EntryPointFitParameters {
    private String modelFilePath;
    private KerasModelType type;
    private String trainFeaturesDirectory;
    private String trainLabelsDirectory;
    private int batchSize;
    private long nbEpoch;
    private String validationXFilePath;
    private String validationYFilePath;
    private String dimOrdering;
}
