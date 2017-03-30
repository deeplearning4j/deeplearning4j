package org.deeplearning4j.keras.api;

import lombok.Builder;
import lombok.Data;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * POJO with parameters of the `fit` method of available through the py4j Python-Java bridge
 */
@Data
@Builder
public class LoadParams {
    private MultiLayerNetwork sequentialModel;
    private ComputationGraph functionalModel;
    private String writePath;
    private boolean saveUpdaterState;
}
