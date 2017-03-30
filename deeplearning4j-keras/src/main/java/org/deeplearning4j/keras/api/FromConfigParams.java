package org.deeplearning4j.keras.api;

import lombok.Builder;
import lombok.Data;
import org.deeplearning4j.keras.model.KerasModelType;

@Data
@Builder
public class FromConfigParams {
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
