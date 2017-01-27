package org.deeplearning4j.keras.api;

import lombok.Builder;
import lombok.Data;
import org.deeplearning4j.keras.model.KerasModelType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

@Data
@Builder
public class EvaluateParams {
    private MultiLayerNetwork sequentialModel;
    private ComputationGraph functionalModel;
    private String featuresDirectory;
    private String labelsDirectory;
    private int batchSize;
}
