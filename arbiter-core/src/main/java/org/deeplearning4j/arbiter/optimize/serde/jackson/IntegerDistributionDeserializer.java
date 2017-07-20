package org.deeplearning4j.arbiter.optimize.serde.jackson;

import org.apache.commons.math3.distribution.*;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;

import java.io.IOException;

/**
 * Custom Jackson deserializer for integer distributions
 *
 * @author Alex Black
 */
public class IntegerDistributionDeserializer extends JsonDeserializer<IntegerDistribution> {

    @Override
    public IntegerDistribution deserialize(JsonParser p, DeserializationContext ctxt) throws IOException {
        JsonNode node = p.getCodec().readTree(p);
        String simpleName = node.get("distribution").asText();

        switch (simpleName) {
            case "BinomialDistribution":
                return new BinomialDistribution(node.get("trials").asInt(), node.get("p").asDouble());
            case "GeometricDistribution":
                return new GeometricDistribution(node.get("p").asDouble());
            case "HypergeometricDistribution":
                return new HypergeometricDistribution(node.get("populationSize").asInt(),
                                node.get("numberOfSuccesses").asInt(), node.get("sampleSize").asInt());
            case "PascalDistribution":
                return new PascalDistribution(node.get("r").asInt(), node.get("p").asDouble());
            case "PoissonDistribution":
                return new PoissonDistribution(node.get("p").asDouble());
            case "UniformIntegerDistribution":
                return new UniformIntegerDistribution(node.get("lower").asInt(), node.get("upper").asInt());
            case "ZipfDistribution":
                return new ZipfDistribution(node.get("numElements").asInt(), node.get("exponent").asDouble());
            default:
                throw new RuntimeException("Unknown or not supported distribution: " + simpleName);
        }
    }
}
