package org.nd4j.tensorflow.conversion.graphrunner;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SavedModelConfig {
    private String savedModelPath, modelTag,signatureKey;
    private List<String> savedModelInputOrder,saveModelOutputOrder;
}
