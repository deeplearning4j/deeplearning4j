package org.deeplearning4j.nn.conf.serde.legacyformat;

import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;

/**
 * Simple helper class to redirect legacy JSON format to {@link LegacyPreprocessorDeserializer} via annotation
 * on {@link org.deeplearning4j.nn.conf.InputPreProcessor}
 */
@JsonDeserialize(using = LegacyPreprocessorDeserializer.class)
public class LegacyPreprocessorDeserializerHelper {
    private LegacyPreprocessorDeserializerHelper(){ }
}
