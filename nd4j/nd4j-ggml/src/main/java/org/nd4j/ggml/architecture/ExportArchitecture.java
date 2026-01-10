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

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.export.ExportOptions;
import org.nd4j.ggml.format.GGMLDataType;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Interface for architecture-specific export logic.
 * Provides name mapping from SameDiff variable names to GGML tensor names
 * and builds GGUF metadata for the target architecture.
 */
public interface ExportArchitecture {

    /**
     * Get the architecture name for GGUF metadata (e.g., "llama", "bert", "gpt2")
     */
    String getArchitectureName();

    /**
     * Get the set of SameDiff variable name patterns that indicate this architecture
     */
    Set<String> getRecognizedVariablePatterns();

    /**
     * Check if this architecture can handle the given SameDiff model
     *
     * @param model the SameDiff model to check
     * @return true if this architecture can handle the model
     */
    boolean canHandle(SameDiff model);

    /**
     * Map a SameDiff variable name to a GGML tensor name
     *
     * @param sdVarName the SameDiff variable name
     * @return the corresponding GGML tensor name, or null if no mapping exists
     */
    String mapVariableName(String sdVarName);

    /**
     * Get the name mapping patterns from SameDiff to GGML
     * Keys are regex patterns for SameDiff names
     * Values are GGML name templates (with {layer} placeholder)
     */
    Map<String, String> getNameMappingPatterns();

    /**
     * Build GGUF metadata for this architecture from the model
     *
     * @param model   the SameDiff model
     * @param options export options
     * @return map of GGUF metadata key-value pairs
     */
    Map<String, Object> buildMetadata(SameDiff model, ExportOptions options);

    /**
     * Get the recommended quantization type for a specific tensor.
     * Allows different quantization for different tensor types
     * (e.g., keep embeddings in FP16, quantize FFN to Q4_K)
     *
     * @param tensorName the GGML tensor name
     * @param options    export options (for default quantization)
     * @return the recommended GGML data type for this tensor
     */
    default GGMLDataType getRecommendedQuantType(String tensorName, ExportOptions options) {
        return options.getQuantizationType().toGGMLDataType();
    }

    /**
     * Validate the model structure for export.
     * Returns a list of validation errors, or empty list if valid.
     *
     * @param model the SameDiff model to validate
     * @return list of validation error messages (empty if valid)
     */
    List<String> validateForExport(SameDiff model);

    /**
     * Get the architecture configuration inferred from the model
     *
     * @param model the SameDiff model
     * @return architecture configuration
     */
    ArchitectureConfig inferConfig(SameDiff model);
}
