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

package org.nd4j.ggml.quantization;

import lombok.Builder;
import lombok.Data;
import org.nd4j.ggml.export.ExportOptions;

import java.util.Arrays;
import java.util.List;

/**
 * Configuration for dynamic per-layer quantization.
 * Allows specifying a target model size and sensitivity metric so that
 * more sensitive layers receive higher-precision quantization while
 * less sensitive layers are quantized more aggressively to meet the
 * size budget.
 */
@Data
@Builder
public class DynamicQuantConfig {

    /**
     * Sensitivity metric used to rank tensors for quantization decisions.
     */
    public enum SensitivityMetric {
        /** Signal-to-noise ratio */
        SNR,
        /** Mean absolute error */
        MAE,
        /** Weight magnitude (L2 norm based) */
        WEIGHT_MAGNITUDE
    }

    /**
     * Target output model size in megabytes.
     * The analyzer will allocate quantization types to meet this budget.
     */
    private long targetModelSizeMB;

    /**
     * Minimum quality quantization type (most aggressive compression).
     * Tensors with lowest sensitivity will use this type.
     */
    @Builder.Default
    private ExportOptions.QuantizationType minQuantType = ExportOptions.QuantizationType.Q4_0;

    /**
     * Maximum quality quantization type (least aggressive compression).
     * Tensors with highest sensitivity will use this type.
     */
    @Builder.Default
    private ExportOptions.QuantizationType maxQuantType = ExportOptions.QuantizationType.Q8_0;

    /**
     * Metric used to measure tensor sensitivity for quantization decisions.
     */
    @Builder.Default
    private SensitivityMetric sensitivityMetric = SensitivityMetric.WEIGHT_MAGNITUDE;

    /**
     * Candidate quantization types to consider during analysis.
     * The analyzer will choose among these types for each tensor.
     */
    @Builder.Default
    private List<ExportOptions.QuantizationType> candidateTypes = Arrays.asList(
            ExportOptions.QuantizationType.Q4_0,
            ExportOptions.QuantizationType.Q4_K,
            ExportOptions.QuantizationType.Q5_K,
            ExportOptions.QuantizationType.Q6_K,
            ExportOptions.QuantizationType.Q8_0
    );

    /**
     * Quantization type used for embedding tensors.
     * Embeddings are typically more sensitive to quantization noise.
     */
    @Builder.Default
    private ExportOptions.QuantizationType embeddingPrecision = ExportOptions.QuantizationType.F16;

    /**
     * Quantization type used for attention layer tensors.
     * Attention weights benefit from higher precision to preserve model quality.
     */
    @Builder.Default
    private ExportOptions.QuantizationType attentionPrecision = ExportOptions.QuantizationType.Q6_K;

    // ========== Factory Methods ==========

    /**
     * Create a dynamic quantization config following Unsloth-style dynamic 2-bit-class strategy.
     * Uses weight magnitude sensitivity, with F16 embeddings and Q6_K attention,
     * allowing the rest to range from Q4_0 to Q8_0.
     *
     * @return a DynamicQuantConfig with Unsloth-inspired defaults
     */
    public static DynamicQuantConfig unslothDynamic2() {
        return DynamicQuantConfig.builder()
                .targetModelSizeMB(0)
                .minQuantType(ExportOptions.QuantizationType.Q4_0)
                .maxQuantType(ExportOptions.QuantizationType.Q8_0)
                .sensitivityMetric(SensitivityMetric.WEIGHT_MAGNITUDE)
                .candidateTypes(Arrays.asList(
                        ExportOptions.QuantizationType.Q4_0,
                        ExportOptions.QuantizationType.Q4_K,
                        ExportOptions.QuantizationType.Q5_K,
                        ExportOptions.QuantizationType.Q6_K,
                        ExportOptions.QuantizationType.Q8_0
                ))
                .embeddingPrecision(ExportOptions.QuantizationType.F16)
                .attentionPrecision(ExportOptions.QuantizationType.Q6_K)
                .build();
    }

    /**
     * Create a standard dynamic quantization config.
     * Uses Q4_K to Q6_K range with weight magnitude sensitivity.
     *
     * @return a DynamicQuantConfig with standard defaults
     */
    public static DynamicQuantConfig standard() {
        return DynamicQuantConfig.builder()
                .targetModelSizeMB(0)
                .minQuantType(ExportOptions.QuantizationType.Q4_K)
                .maxQuantType(ExportOptions.QuantizationType.Q6_K)
                .sensitivityMetric(SensitivityMetric.WEIGHT_MAGNITUDE)
                .candidateTypes(Arrays.asList(
                        ExportOptions.QuantizationType.Q4_K,
                        ExportOptions.QuantizationType.Q5_K,
                        ExportOptions.QuantizationType.Q6_K
                ))
                .embeddingPrecision(ExportOptions.QuantizationType.F16)
                .attentionPrecision(ExportOptions.QuantizationType.Q6_K)
                .build();
    }
}
