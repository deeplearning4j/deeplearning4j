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

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Configuration for per-layer adaptive GGUF quantization.
 *
 * <p>Adaptive quantization (inspired by Unsloth Dynamic 2.0) measures each layer's
 * sensitivity to quantization noise via KL divergence between full-precision and
 * quantized layer outputs on a calibration dataset. Layers in the top
 * {@link #highSensitivityPercentile} by sensitivity score are assigned the highest
 * quality type; layers in the bottom {@link #lowSensitivityPercentile} receive the
 * most aggressive type. The remaining middle layers are distributed evenly across
 * the remaining candidate types.
 *
 * <p>Example:
 * <pre>{@code
 * AdaptiveQuantConfig config = AdaptiveQuantConfig.balanced();
 * AdaptiveLayerQuantizer quantizer = new AdaptiveLayerQuantizer();
 * Map<String, GGMLQuantType> plan = quantizer.analyzeAndAssign(weights, calibData, config);
 * }</pre>
 *
 * @see AdaptiveLayerQuantizer
 */
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class AdaptiveQuantConfig {

    /**
     * Candidate quantization types to choose from, ordered from highest quality
     * (most bits) to lowest quality (fewest bits). The first entry is assigned to
     * the most sensitive layers; the last entry is assigned to the least sensitive.
     */
    @Builder.Default
    private List<GGMLQuantType> quantTypes = Arrays.asList(
            GGMLQuantType.Q6_K,
            GGMLQuantType.Q5_K,
            GGMLQuantType.Q4_K,
            GGMLQuantType.Q3_K,
            GGMLQuantType.Q2_K
    );

    /**
     * Target average bits per weight across all quantized layers.
     * The quantizer will try to match this budget while honouring sensitivity ordering.
     * Default: 4.0 (roughly Q4_K equivalent).
     *
     * <p>Set to 0 to disable budget enforcement and use pure rank-based assignment.
     */
    @Builder.Default
    private double targetBitsPerWeight = 4.0;

    /**
     * Top fraction of layers (by sensitivity score) that receive the highest
     * quality quantization type. Must be in [0, 1). Default: 0.25.
     */
    @Builder.Default
    private double highSensitivityPercentile = 0.25;

    /**
     * Bottom fraction of layers (by sensitivity score) that receive the most
     * aggressive quantization type. Must be in [0, 1). Default: 0.25.
     */
    @Builder.Default
    private double lowSensitivityPercentile = 0.25;

    /**
     * Number of calibration samples used to compute KL divergence sensitivities.
     * Larger values are more accurate but slower. Default: 128.
     */
    @Builder.Default
    private int numCalibrationSamples = 128;

    /**
     * Sequence length used for calibration inputs. Default: 2048.
     */
    @Builder.Default
    private int calibrationSeqLen = 2048;

    /**
     * Layers that must always be kept at the highest precision regardless of
     * their measured sensitivity. Typical candidates are embeddings and the
     * unembedding / language-model head.
     */
    @Builder.Default
    private Set<String> preserveLayers = new HashSet<>(Arrays.asList(
            "output.weight",
            "token_embd.weight"
    ));

    // ========== Validation ==========

    /**
     * Validate configuration parameters.
     *
     * @throws IllegalArgumentException if any parameter is out of range
     */
    public void validate() {
        if (targetBitsPerWeight < 0) {
            throw new IllegalArgumentException(
                    "targetBitsPerWeight must be >= 0, got: " + targetBitsPerWeight);
        }
        if (targetBitsPerWeight > 32) {
            throw new IllegalArgumentException(
                    "targetBitsPerWeight must be <= 32, got: " + targetBitsPerWeight);
        }
        if (highSensitivityPercentile < 0 || highSensitivityPercentile >= 1.0) {
            throw new IllegalArgumentException(
                    "highSensitivityPercentile must be in [0, 1), got: " + highSensitivityPercentile);
        }
        if (lowSensitivityPercentile < 0 || lowSensitivityPercentile >= 1.0) {
            throw new IllegalArgumentException(
                    "lowSensitivityPercentile must be in [0, 1), got: " + lowSensitivityPercentile);
        }
        if (highSensitivityPercentile + lowSensitivityPercentile >= 1.0) {
            throw new IllegalArgumentException(
                    "highSensitivityPercentile + lowSensitivityPercentile must be < 1.0, got: "
                            + (highSensitivityPercentile + lowSensitivityPercentile));
        }
        if (numCalibrationSamples <= 0) {
            throw new IllegalArgumentException(
                    "numCalibrationSamples must be > 0, got: " + numCalibrationSamples);
        }
        if (calibrationSeqLen <= 0) {
            throw new IllegalArgumentException(
                    "calibrationSeqLen must be > 0, got: " + calibrationSeqLen);
        }
        if (quantTypes == null || quantTypes.isEmpty()) {
            throw new IllegalArgumentException("quantTypes must not be null or empty");
        }
    }

    /**
     * Return the highest quality type in {@link #quantTypes} (first in ordered list).
     */
    public GGMLQuantType highestQualityType() {
        return quantTypes.get(0);
    }

    /**
     * Return the lowest quality type in {@link #quantTypes} (last in ordered list).
     */
    public GGMLQuantType lowestQualityType() {
        return quantTypes.get(quantTypes.size() - 1);
    }

    // ========== Factory Methods ==========

    /**
     * Balanced preset: Q6_K – Q4_K range, ~4.5 bpw target.
     * Good trade-off between quality and model size for most use cases.
     */
    public static AdaptiveQuantConfig balanced() {
        return AdaptiveQuantConfig.builder()
                .quantTypes(Arrays.asList(
                        GGMLQuantType.Q6_K,
                        GGMLQuantType.Q5_K,
                        GGMLQuantType.Q4_K))
                .targetBitsPerWeight(4.5)
                .highSensitivityPercentile(0.25)
                .lowSensitivityPercentile(0.25)
                .preserveLayers(new HashSet<>(Arrays.asList("output.weight", "token_embd.weight")))
                .build();
    }

    /**
     * Quality preset: Q6_K – Q5_K range, ~6.0 bpw target.
     * Prioritises model accuracy; suitable when disk space is not constrained.
     */
    public static AdaptiveQuantConfig quality() {
        return AdaptiveQuantConfig.builder()
                .quantTypes(Arrays.asList(
                        GGMLQuantType.Q6_K,
                        GGMLQuantType.Q5_K))
                .targetBitsPerWeight(6.0)
                .highSensitivityPercentile(0.30)
                .lowSensitivityPercentile(0.20)
                .preserveLayers(new HashSet<>(Arrays.asList("output.weight", "token_embd.weight")))
                .build();
    }

    /**
     * Aggressive preset: Q4_K – Q2_K range, ~3.0 bpw target.
     * Maximum compression; expect some quality degradation.
     */
    public static AdaptiveQuantConfig aggressive() {
        return AdaptiveQuantConfig.builder()
                .quantTypes(Arrays.asList(
                        GGMLQuantType.Q4_K,
                        GGMLQuantType.Q3_K,
                        GGMLQuantType.Q2_K))
                .targetBitsPerWeight(3.0)
                .highSensitivityPercentile(0.20)
                .lowSensitivityPercentile(0.30)
                .preserveLayers(new HashSet<>(Collections.singletonList("token_embd.weight")))
                .build();
    }
}
