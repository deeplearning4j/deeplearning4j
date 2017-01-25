package org.deeplearning4j.keras.api.sequential;

import lombok.Builder;
import lombok.Data;
import org.deeplearning4j.keras.model.KerasModelType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * POJO with parameters of the `fit` method of available through the py4j Python-Java bridge
 */
@Data
@Builder
public class FitParams {
    private MultiLayerNetwork model;
    private String trainFeaturesDirectory;
    private String trainLabelsDirectory;
    private int batchSize;
    private long nbEpoch;
    private String validationXFilePath;
    private String validationYFilePath;
    private String dimOrdering;
}
