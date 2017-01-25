package org.deeplearning4j.keras.api.sequential;

import lombok.Builder;
import lombok.Data;
import org.deeplearning4j.keras.model.KerasModelType;

@Data
@Builder
public class SequentialModelRef {
    private String modelFilePath;
    private KerasModelType type;
}
