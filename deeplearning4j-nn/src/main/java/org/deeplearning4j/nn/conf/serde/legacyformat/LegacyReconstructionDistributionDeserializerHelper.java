package org.deeplearning4j.nn.conf.serde.legacyformat;

import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;

/**
 * Simple helper class to redirect legacy JSON format to {@link LegacyReconstructionDistributionDeserializer} via annotation
 * on {@link org.deeplearning4j.nn.conf.layers.variational.ReconstructionDistribution}
 */
@JsonDeserialize(using = LegacyReconstructionDistributionDeserializer.class)
public class LegacyReconstructionDistributionDeserializerHelper {
    private LegacyReconstructionDistributionDeserializerHelper(){ }
}
