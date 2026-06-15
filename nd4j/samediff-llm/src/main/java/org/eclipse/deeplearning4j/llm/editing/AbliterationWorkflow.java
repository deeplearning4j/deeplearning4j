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
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * High-level orchestrator for the abliteration workflow.
 *
 * <p>Abliteration ("ablation + obliteration") is a training-free technique for
 * removing refusal behavior from language models. Based on the NeurIPS 2024 paper
 * "Refusal in Language Models Is Mediated by a Single Direction" (Arditi et al.).</p>
 *
 * <p>The workflow:</p>
 * <ol>
 *   <li>Collect activations from the model on harmful and harmless prompts</li>
 *   <li>Identify the refusal direction in the residual stream</li>
 *   <li>Orthogonalize target weight matrices against the refusal direction</li>
 * </ol>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * AbliterationResult result = AbliterationWorkflow.builder()
 *     .model(sameDiffModel)
 *     .config(AbliterationConfig.builder().method(Method.PROJECTED).build())
 *     .harmfulActivations(harmfulActMap)
 *     .harmlessActivations(harmlessActMap)
 *     .build()
 *     .execute();
 * }</pre>
 *
 * <p>For end-to-end usage with prompt-based activation collection:</p>
 * <pre>{@code
 * AbliterationResult result = AbliterationWorkflow.builder()
 *     .model(sameDiffModel)
 *     .config(AbliterationConfig.defaults())
 *     .harmfulPrompts(DefaultPromptSets.getDefaultHarmfulPrompts())
 *     .harmlessPrompts(DefaultPromptSets.getDefaultHarmlessPrompts())
 *     .modelInputs(inputMap)
 *     .build()
 *     .execute();
 * }</pre>
 */
@Slf4j
@Builder
public class AbliterationWorkflow {

    /** The SameDiff model to modify. */
    private final SameDiff model;

    /** Abliteration configuration. */
    @Builder.Default
    private final AbliterationConfig config = AbliterationConfig.defaults();

    /**
     * Pre-computed activations from harmful prompts.
     * Map from variable name to INDArray of shape [numSamples, hiddenDim].
     * If provided, harmfulPrompts is ignored.
     */
    private final Map<String, INDArray> harmfulActivations;

    /**
     * Pre-computed activations from harmless prompts.
     * Map from variable name to INDArray of shape [numSamples, hiddenDim].
     * If provided, harmlessPrompts is ignored.
     */
    private final Map<String, INDArray> harmlessActivations;

    /**
     * Harmful prompts for activation collection (used when harmfulActivations is null).
     */
    private final List<String> harmfulPrompts;

    /**
     * Harmless prompts for activation collection (used when harmlessActivations is null).
     */
    private final List<String> harmlessPrompts;

    /**
     * Model inputs for forward pass when collecting activations from prompts.
     * Required when using prompt-based activation collection.
     */
    private final Map<String, INDArray> modelInputs;

    /**
     * Execute the abliteration workflow.
     *
     * @return result containing directions found, applied, and weights modified
     */
    public AbliterationResult execute() {
        Preconditions.checkNotNull(model, "Model must not be null");

        Map<String, INDArray> harmfulActs = harmfulActivations;
        Map<String, INDArray> harmlessActs = harmlessActivations;

        // If pre-computed activations not provided, collect from prompts
        if (harmfulActs == null || harmlessActs == null) {
            Preconditions.checkArgument(
                    harmfulPrompts != null && !harmfulPrompts.isEmpty(),
                    "Either harmfulActivations or harmfulPrompts must be provided");
            Preconditions.checkArgument(
                    harmlessPrompts != null && !harmlessPrompts.isEmpty(),
                    "Either harmlessActivations or harmlessPrompts must be provided");
            Preconditions.checkNotNull(modelInputs,
                    "modelInputs must be provided when using prompt-based activation collection");

            RefusalDirectionFinder finder = new RefusalDirectionFinder();
            List<String> probePoints = finder.identifyProbePoints(model, config);
            Preconditions.checkArgument(!probePoints.isEmpty(),
                    "No probe points found in model matching target weight patterns");

            harmfulActs = finder.collectActivations(model, modelInputs, probePoints);
            harmlessActs = finder.collectActivations(model, modelInputs, probePoints);
        }

        // Step 1: Find refusal directions
        log.info("Finding refusal directions using {} method...", config.getMethod());
        RefusalDirectionFinder finder = new RefusalDirectionFinder();
        List<RefusalDirection> allCandidates = finder.findRefusalDirections(
                model, harmfulActs, harmlessActs, config);

        log.info("Found {} candidate directions:", allCandidates.size());
        for (int i = 0; i < Math.min(10, allCandidates.size()); i++) {
            log.info("  #{}: {}", i + 1, allCandidates.get(i));
        }

        // Step 2: Select directions to apply
        List<RefusalDirection> selected = selectDirections(allCandidates);
        log.info("Selected {} direction(s) for orthogonalization", selected.size());

        // Step 3: Orthogonalize weights
        WeightOrthogonalizer orthogonalizer = new WeightOrthogonalizer();
        List<String> modifiedNames = orthogonalizer.orthogonalizeWeights(
                model, selected, config);

        AbliterationResult result = AbliterationResult.builder()
                .allCandidates(allCandidates)
                .appliedDirections(selected)
                .numWeightsModified(modifiedNames.size())
                .modifiedWeightNames(modifiedNames)
                .build();

        log.info("Abliteration complete: {}", result);
        return result;
    }

    /**
     * Select directions to apply based on the layer selection strategy.
     */
    private List<RefusalDirection> selectDirections(List<RefusalDirection> candidates) {
        if (candidates.isEmpty()) {
            return candidates;
        }

        switch (config.getLayerSelectionStrategy()) {
            case BEST_SINGLE:
                List<RefusalDirection> best = new ArrayList<>();
                best.add(candidates.get(0));
                return best;

            case TOP_K:
                int k = Math.min(config.getTopK(), candidates.size());
                return new ArrayList<>(candidates.subList(0, k));

            case ALL_LAYERS:
                return new ArrayList<>(candidates);

            default:
                throw new IllegalArgumentException(
                        "Unknown layer selection strategy: " + config.getLayerSelectionStrategy());
        }
    }
}
