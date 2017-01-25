package org.deeplearning4j.keras.api.sequential;

import lombok.Builder;
import lombok.Data;
import org.deeplearning4j.keras.model.KerasModelType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

@Data
@Builder
public class PredictOnBatchParams {
    private MultiLayerNetwork model;
    private String featuresDirectory;
}
