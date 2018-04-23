package org.deeplearning4j.nn.conf.serde.legacyformat;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.layers.variational.*;
import org.deeplearning4j.nn.conf.preprocessor.*;
import org.deeplearning4j.nn.conf.serde.JsonMappers;
import org.nd4j.serde.json.BaseLegacyDeserializer;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.util.HashMap;
import java.util.Map;

public class LegacyReconstructionDistributionDeserializer extends BaseLegacyDeserializer<ReconstructionDistribution> {

    private static final Map<String,String> LEGACY_NAMES = new HashMap<>();

    static {
        LEGACY_NAMES.put("Gaussian", GaussianReconstructionDistribution.class.getName());
        LEGACY_NAMES.put("Bernoulli", BernoulliReconstructionDistribution.class.getName());
        LEGACY_NAMES.put("Exponential", ExponentialReconstructionDistribution.class.getName());
        LEGACY_NAMES.put("Composite", CompositeReconstructionDistribution.class.getName());
        LEGACY_NAMES.put("LossWrapper", LossFunctionWrapper.class.getName());
    }


    @Override
    public Map<String, String> getLegacyNamesMap() {
        return LEGACY_NAMES;
    }

    @Override
    public ObjectMapper getLegacyJsonMapper() {
        return JsonMappers.getMapperLegacyJson();
    }
}
