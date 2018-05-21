package org.deeplearning4j.nn.conf.distribution.serde;

import org.deeplearning4j.nn.conf.distribution.*;
import org.nd4j.shade.jackson.core.JsonParseException;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;

import java.io.IOException;

/**
 * Jackson Json deserializer to handle legacy format for distributions.<br>
 * Now, we use 'type' field which contains class information.<br>
 * Previously, we used wrapper objects for type information instead (see TestDistributionDeserializer for examples)
 *
 * @author Alex Black
 */
public class LegacyDistributionDeserializer extends JsonDeserializer<Distribution> {
    @Override
    public Distribution deserialize(JsonParser jp, DeserializationContext deserializationContext)
                    throws IOException, JsonProcessingException {
        //Manually parse old format
        JsonNode node = jp.getCodec().readTree(jp);

        if (node.has("normal")) {
            JsonNode n = node.get("normal");
            if (!n.has("mean") || !n.has("std")) {
                throw new JsonParseException("Cannot deserialize Distribution: legacy format 'normal' wrapper object "
                                + " is missing 'mean' or 'std' field", jp.getCurrentLocation());
            }
            double m = n.get("mean").asDouble();
            double s = n.get("std").asDouble();
            return new NormalDistribution(m, s);
        } else if (node.has("gaussian")) {
            JsonNode n = node.get("gaussian");
            if (!n.has("mean") || !n.has("std")) {
                throw new JsonParseException("Cannot deserialize Distribution: legacy format 'gaussian' wrapper object "
                                + " is missing 'mean' or 'std' field", jp.getCurrentLocation());
            }
            double m = n.get("mean").asDouble();
            double s = n.get("std").asDouble();
            return new GaussianDistribution(m, s);

        } else if (node.has("uniform")) {
            JsonNode n = node.get("uniform");
            if (!n.has("lower") || !n.has("upper")) {
                throw new JsonParseException("Cannot deserialize Distribution: legacy format 'uniform' wrapper object "
                                + " is missing 'lower' or 'upper' field", jp.getCurrentLocation());
            }
            double l = n.get("lower").asDouble();
            double u = n.get("upper").asDouble();
            return new UniformDistribution(l, u);
        } else if (node.has("binomial")) {
            JsonNode n = node.get("binomial");
            if (!n.has("numberOfTrials") || !n.has("probabilityOfSuccess")) {
                throw new JsonParseException("Cannot deserialize Distribution: legacy format 'binomial' wrapper object "
                                + " is missing 'lower' or 'upper' field", jp.getCurrentLocation());
            }
            int num = n.get("numberOfTrials").asInt();
            double p = n.get("probabilityOfSuccess").asDouble();
            return new BinomialDistribution(num, p);
        } else {
            throw new JsonParseException(
                            "Cannot deserialize Distribution: expected type field or legacy format wrapper"
                                            + " object with name being one of {normal, gaussian, uniform, binomial}",
                            jp.getCurrentLocation());
        }
    }
}
