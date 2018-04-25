package org.deeplearning4j.nn.conf.serde.legacyformat;

import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;

@JsonDeserialize(using = LegacyReconstructionDistributionDeserializer.class)
public class LegacyReconstructionDistributionDeserializerHelper {
    private LegacyReconstructionDistributionDeserializerHelper(){ }
}
