package org.deeplearning4j.arbiter.optimize.serde.jackson;

import org.apache.commons.math3.distribution.*;
import org.deeplearning4j.arbiter.optimize.distribution.LogUniformDistribution;
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
    public RealDistribution deserialize(JsonParser p, DeserializationContext ctxt)
                    throws IOException, JsonProcessingException {
        JsonNode node = p.getCodec().readTree(p);
        String simpleName = node.get("distribution").asText();

        switch (simpleName) {
            case "BetaDistribution":
                return new BetaDistribution(node.get("alpha").asDouble(), node.get("beta").asDouble());
            case "CauchyDistribution":
                return new CauchyDistribution(node.get("median").asDouble(), node.get("scale").asDouble());
            case "ChiSquaredDistribution":
                return new ChiSquaredDistribution(node.get("dof").asDouble());
            case "ExponentialDistribution":
                return new ExponentialDistribution(node.get("mean").asDouble());
            case "FDistribution":
                return new FDistribution(node.get("numeratorDof").asDouble(), node.get("denominatorDof").asDouble());
            case "GammaDistribution":
                return new GammaDistribution(node.get("shape").asDouble(), node.get("scale").asDouble());
            case "LevyDistribution":
                return new LevyDistribution(node.get("mu").asDouble(), node.get("c").asDouble());
            case "LogNormalDistribution":
                return new LogNormalDistribution(node.get("scale").asDouble(), node.get("shape").asDouble());
            case "NormalDistribution":
                return new NormalDistribution(node.get("mean").asDouble(), node.get("stdev").asDouble());
            case "ParetoDistribution":
                return new ParetoDistribution(node.get("scale").asDouble(), node.get("shape").asDouble());
            case "TDistribution":
                return new TDistribution(node.get("dof").asDouble());
            case "TriangularDistribution":
                return new TriangularDistribution(node.get("a").asDouble(), node.get("b").asDouble(),
                                node.get("c").asDouble());
            case "UniformRealDistribution":
                return new UniformRealDistribution(node.get("lower").asDouble(), node.get("upper").asDouble());
            case "WeibullDistribution":
                return new WeibullDistribution(node.get("alpha").asDouble(), node.get("beta").asDouble());
            case "LogUniformDistribution":
                return new LogUniformDistribution(node.get("min").asDouble(), node.get("max").asDouble());
            default:
                throw new RuntimeException("Unknown or not supported distribution: " + simpleName);
        }


    }
}
