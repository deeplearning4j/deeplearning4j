package org.deeplearning4j.nn.conf.serde.legacyformat;

import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;

@JsonDeserialize(using = LegacyGraphVertexDeserializer.class)
public class LegacyGraphVertexDeserializerHelper {
    private LegacyGraphVertexDeserializerHelper(){ }
}
