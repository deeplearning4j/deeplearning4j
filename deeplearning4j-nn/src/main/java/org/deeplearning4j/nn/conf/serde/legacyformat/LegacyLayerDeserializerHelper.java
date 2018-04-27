package org.deeplearning4j.nn.conf.serde.legacyformat;

import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;

/**
 * Simple helper class to redirect legacy JSON format to {@link LegacyLayerDeserializer} via annotation
 * on {@link org.deeplearning4j.nn.conf.layers.Layer}
 */
@JsonDeserialize(using = LegacyLayerDeserializer.class)
public class LegacyLayerDeserializerHelper {
    private LegacyLayerDeserializerHelper(){ }
}
