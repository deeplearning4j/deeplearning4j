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

package org.eclipse.deeplearning4j.llm.editing;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Configuration for abliteration (training-free refusal removal).
 *
 * <p>Based on "Refusal in Language Models Is Mediated by a Single Direction"
 * (Arditi et al., NeurIPS 2024). Abliteration identifies a linear direction in
 * the residual stream that encodes refusal behavior and removes it via
 * weight orthogonalization.</p>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * AbliterationConfig config = AbliterationConfig.builder()
 *     .method(AbliterationConfig.Method.PROJECTED)
 *     .layerSelectionStrategy(AbliterationConfig.LayerSelection.BEST_SINGLE)
 *     .ablationStrength(0.8)
 *     .build();
 * }</pre>
 */
@Data
@Builder
public class AbliterationConfig {

    /**
     * Method for computing the refusal direction.
     */
    public enum Method {
        /** Mean difference between harmful and harmless activations (Arditi et al.) */
        DIFF_IN_MEANS,
        /** Top principal component of the difference matrix */
        PCA,
        /** Projected variant — only the component orthogonal to harmless mean (grimjim) */
        PROJECTED
    }

    /**
     * Strategy for selecting which layer directions to ablate.
     */
    public enum LayerSelection {
        /** Ablate only the single highest-scoring direction */
        BEST_SINGLE,
        /** Ablate the top-K highest-scoring directions */
        TOP_K,
        /** Ablate directions from all layers */
        ALL_LAYERS
    }

    /**
     * Positions in the residual stream to probe for refusal directions.
     */
    public enum ResidualStreamPosition {
        /** Before the attention block (layer norm input) */
        PRE_ATTENTION,
        /** After attention output projection */
        POST_ATTENTION,
        /** After MLP output projection */
        POST_MLP
    }

    /** Number of harmful prompts to use for direction computation. */
    @Builder.Default
    private int numHarmfulSamples = 256;

    /** Number of harmless prompts to use for direction computation. */
    @Builder.Default
    private int numHarmlessSamples = 256;

    /** Method for computing the refusal direction. */
    @Builder.Default
    private Method method = Method.DIFF_IN_MEANS;

    /** Strategy for selecting which layer directions to ablate. */
    @Builder.Default
    private LayerSelection layerSelectionStrategy = LayerSelection.BEST_SINGLE;

    /** Number of directions to ablate when using TOP_K strategy. */
    @Builder.Default
    private int topK = 1;

    /** Scaling factor for ablation strength (1.0 = full removal). */
    @Builder.Default
    private double ablationStrength = 1.0;

    /** Percentile for Winsorization of activation outliers (e.g., for Gemma). */
    @Builder.Default
    private double winsorizePercentile = 0.995;

    /** Whether to apply Winsorization to activations before direction computation. */
    @Builder.Default
    private boolean winsorize = false;

    /** Data type for direction computation. */
    @Builder.Default
    private DataType computeDataType = DataType.FLOAT;

    /** Residual stream positions to probe. */
    @Builder.Default
    private Set<ResidualStreamPosition> residualStreamPositions = EnumSet.of(
            ResidualStreamPosition.POST_ATTENTION,
            ResidualStreamPosition.POST_MLP
    );

    /**
     * Regex patterns for weight variable names to orthogonalize.
     * Defaults cover common architectures (Llama, Mistral, Gemma, Phi, Qwen).
     */
    @Builder.Default
    private List<String> targetWeightPatterns = Arrays.asList(
            // Embedding matrices
            ".*embed_tokens.*",
            ".*wte.*",
            ".*word_embeddings.*",
            // Attention output projections
            ".*\\.o_proj\\.weight.*",
            ".*\\.out_proj\\.weight.*",
            ".*\\.attn\\.c_proj.*",
            // MLP output projections
            ".*\\.down_proj\\.weight.*",
            ".*\\.fc_out\\.weight.*",
            ".*\\.fc2\\.weight.*",
            ".*\\.mlp\\.c_proj.*",
            // LM head
            ".*lm_head.*"
    );

    /**
     * Create a default configuration suitable for most Llama-family models.
     */
    public static AbliterationConfig defaults() {
        return AbliterationConfig.builder().build();
    }

    /**
     * Create a configuration using the projected method with conservative strength.
     * Recommended for minimizing collateral damage to model capabilities.
     */
    public static AbliterationConfig conservative() {
        return AbliterationConfig.builder()
                .method(Method.PROJECTED)
                .ablationStrength(0.8)
                .build();
    }

    /**
     * Create a configuration with Winsorization enabled, suitable for
     * architectures with activation outliers (e.g., Gemma).
     */
    public static AbliterationConfig withWinsorization() {
        return AbliterationConfig.builder()
                .winsorize(true)
                .winsorizePercentile(0.995)
                .build();
    }
}
