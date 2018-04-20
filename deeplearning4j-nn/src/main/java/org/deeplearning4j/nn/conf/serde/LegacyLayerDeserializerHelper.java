package org.deeplearning4j.nn.conf.serde;

import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;

@JsonDeserialize(using = LegacyLayerDeserializer.class)
public class LegacyLayerDeserializerHelper {
    private LegacyLayerDeserializerHelper(){ }
}
