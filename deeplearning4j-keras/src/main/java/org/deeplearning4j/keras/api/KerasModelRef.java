package org.deeplearning4j.keras.api;

import lombok.Builder;
import lombok.Data;
import org.deeplearning4j.keras.model.KerasModelType;

@Data
@Builder
public class KerasModelRef {
    private String modelFilePath;
    private KerasModelType type;
}
