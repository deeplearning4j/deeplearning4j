/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

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
