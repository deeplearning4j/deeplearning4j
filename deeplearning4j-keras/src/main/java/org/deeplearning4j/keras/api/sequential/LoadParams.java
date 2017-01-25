package org.deeplearning4j.keras.api.sequential;

import lombok.Builder;
import lombok.Data;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * POJO with parameters of the `fit` method of available through the py4j Python-Java bridge
 */
@Data
@Builder
public class LoadParams {
    private MultiLayerNetwork model;
    private String writePath;
    private boolean saveUpdaterState;
}
