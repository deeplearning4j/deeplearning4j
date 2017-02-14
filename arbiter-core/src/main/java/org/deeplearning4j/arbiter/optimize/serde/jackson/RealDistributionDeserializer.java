package org.deeplearning4j.arbiter.optimize.serde.jackson;

import org.apache.commons.math3.distribution.LogNormalDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;

import java.io.IOException;

/**
 * Created by Alex on 14/02/2017.
 */
public class RealDistributionDeserializer extends JsonDeserializer<RealDistribution> {

    @Override
    public RealDistribution deserialize(JsonParser p, DeserializationContext ctxt) throws IOException, JsonProcessingException {
        JsonNode node = p.getCodec().readTree(p);
        String simpleName = node.get("distribution").asText();

        switch (simpleName){
            case "BetaDistribution":
                throw new UnsupportedOperationException("Not yet implemented");
            case "CauchyDistribution":
                throw new UnsupportedOperationException("Not yet implemented");
            case "ChiSquaredDistribution":
                throw new UnsupportedOperationException("Not yet implemented");
            case "ExponentialDistribution":
                throw new UnsupportedOperationException("Not yet implemented");
            case "FDistribution":
                throw new UnsupportedOperationException("Not yet implemented");
            case "GammaDistribution":
                throw new UnsupportedOperationException("Not yet implemented");
            case "LevyDistribution":
                throw new UnsupportedOperationException("Not yet implemented");
            case "LogNormalDistribution":
                return new LogNormalDistribution(node.get("scale").asDouble(), node.get("shape").asDouble());
            case "NormalDistribution":
                return new NormalDistribution(node.get("mean").asDouble(), node.get("stdev").asDouble());
            case "ParetoDistribution":
                throw new UnsupportedOperationException("Not yet implemented");
            case "TDistribution":
                throw new UnsupportedOperationException("Not yet implemented");
            case "TriangularDistribution":
                throw new UnsupportedOperationException("Not yet implemented");
            case "UniformRealDistribution":
                return new UniformRealDistribution(node.get("lower").asDouble(), node.get("upper").asDouble());
            case "WeibullDistribution":
                throw new UnsupportedOperationException("Not yet implemented");
            default:
                throw new RuntimeException("Unknown or not supported distribution: " + simpleName);
        }


    }
}
