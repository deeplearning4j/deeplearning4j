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

package org.eclipse.deeplearning4j.safetensors.architecture;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;
import java.util.Set;

/**
 * Interface for SafeTensors model architecture builders.
 *
 * <p>Each implementation handles a specific model architecture and knows how to
 * build a SameDiff compute graph from SafeTensors weights and a HuggingFace config.json.</p>
 *
 * <p>This mirrors the GGUF {@code ModelArchitecture} pattern but for SafeTensors format,
 * where weight names follow HuggingFace conventions (e.g., {@code model.layers.0.self_attn.q_proj.weight})
 * rather than GGUF conventions (e.g., {@code blk.0.attn_q.weight}).</p>
 */
public interface SafeTensorsArchitecture {

    /**
     * Get the primary name of this architecture.
     */
    String getName();

    /**
     * Get all architecture identifiers this builder handles.
     * These are matched against the "architectures" field in config.json.
     */
    Set<String> getSupportedArchitectures();

    /**
     * Check if this builder can handle a model with the given config.
     *
     * @param config parsed config.json as a map
     * @return true if this builder can construct a graph for this config
     */
    boolean canHandle(Map<String, Object> config);

    /**
     * Build a SameDiff compute graph from weights and configuration.
     *
     * @param weights map of SafeTensors weight names to INDArray values
     * @param config parsed config.json as a map
     * @return the constructed SameDiff graph with weights as constants
     */
    SameDiff buildGraph(Map<String, INDArray> weights, Map<String, Object> config);

    /**
     * Get the HuggingFace weight name to SameDiff variable name mappings.
     * Uses {layer} as a placeholder for layer indices.
     */
    Map<String, String> getWeightNameMappings();
}
