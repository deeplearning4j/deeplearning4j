/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.ggml.architecture;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

/**
 * Generic architecture handler that loads weights without building a specific computation graph.
 * Useful for architectures that don't have a specific handler yet.
 */
@Slf4j
public class GenericArchitecture implements ModelArchitecture {

    @Override
    public String getName() {
        return "generic";
    }

    @Override
    public Set<String> getSupportedVariants() {
        return Set.of("generic", "unknown");
    }

    @Override
    public boolean canHandle(GGMLMetadata metadata) {
        // Generic handler can handle anything as a fallback
        return true;
    }

    @Override
    public SameDiff buildGraph(GGMLMetadata metadata, Map<String, INDArray> weights, ConversionOptions options) {
        SameDiff sd = SameDiff.create();

        log.info("Building generic graph with {} tensors", weights.size());

        // Simply store all weights as variables
        for (Map.Entry<String, INDArray> entry : weights.entrySet()) {
            String name = entry.getKey();
            INDArray weight = entry.getValue();

            // Clean up name for SameDiff variable naming
            String varName = name.replace(".", "_").replace("-", "_");

            if (options.isForTraining()) {
                // Create as trainable variable
                sd.var(varName, weight);
            } else {
                // Create as constant for inference
                sd.constant(varName, weight);
            }

            log.debug("Added variable: {} with shape {}", varName, weight.shapeInfoToString());
        }

        return sd;
    }

    @Override
    public Map<String, String> getTensorNamePatterns() {
        // No specific patterns for generic architecture
        return Collections.emptyMap();
    }
}
