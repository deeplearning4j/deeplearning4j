package org.deeplearning4j.nn.conf.serde.legacyformat;

import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;

/**
 * Simple helper class to redirect legacy JSON format to {@link LegacyGraphVertexDeserializer} via annotation
 * on {@link org.deeplearning4j.nn.conf.graph.GraphVertex}
 */
@JsonDeserialize(using = LegacyGraphVertexDeserializer.class)
public class LegacyGraphVertexDeserializerHelper {
    private LegacyGraphVertexDeserializerHelper(){ }
}
