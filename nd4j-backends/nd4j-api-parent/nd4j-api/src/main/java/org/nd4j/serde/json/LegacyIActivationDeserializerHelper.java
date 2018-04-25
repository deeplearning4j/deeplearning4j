package org.nd4j.serde.json;

import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;

/**
 * Simple helper class to redirect legacy JSON format to {@link LegacyIActivationDeserializer} via annotation
 * on {@link org.nd4j.linalg.activations.IActivation}
 */
@JsonDeserialize(using = LegacyIActivationDeserializer.class)
public class LegacyIActivationDeserializerHelper {
    private LegacyIActivationDeserializerHelper(){ }
}
