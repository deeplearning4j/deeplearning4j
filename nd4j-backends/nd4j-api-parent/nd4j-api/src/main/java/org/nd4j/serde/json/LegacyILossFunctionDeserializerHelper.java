package org.nd4j.serde.json;

import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;

/**
 * Simple helper class to redirect legacy JSON format to {@link LegacyILossFunctionDeserializer} via annotation
 * on {@link org.nd4j.linalg.lossfunctions.ILossFunction}
 */
@JsonDeserialize(using = LegacyILossFunctionDeserializer.class)
public class LegacyILossFunctionDeserializerHelper {
    private LegacyILossFunctionDeserializerHelper(){ }
}
