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

package org.eclipse.deeplearning4j.vlm.model;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * DeepStack feature merger for Qwen3-VL.
 *
 * <p>DeepStack extracts intermediate features from specific ViT layers
 * and routes them to the language model via lightweight residual connections.
 * This provides multi-scale visual information to the LLM.</p>
 *
 * <p>For Qwen3-VL, features are extracted at ViT layers [8, 16, 24] and
 * routed to corresponding early, middle, and late LLM layers.</p>
 *
 * <p>The merge operation for each level:</p>
 * <pre>
 * llm_hidden = llm_hidden + alpha * project(vit_hidden)
 * </pre>
 * <p>where alpha is a learnable scalar and project() is a linear projection
 * from vision hidden size to LLM hidden size.</p>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * DeepStackMerger merger = DeepStackMerger.forQwen3VL();
 *
 * // Collect intermediate ViT outputs during encoding
 * Map<Integer, INDArray> vitFeatures = new HashMap<>();
 * vitFeatures.put(8, vitLayer8Output);
 * vitFeatures.put(16, vitLayer16Output);
 * vitFeatures.put(24, vitLayer24Output);
 *
 * // Merge into LLM hidden states
 * INDArray mergedHidden = merger.merge(llmHidden, vitFeatures);
 * }</pre>
 *
 * @see VisionEncoder
 */
@Slf4j
@Builder
@Getter
public class DeepStackMerger {

    @Builder.Default
    private int[] visualIndexes = {8, 16, 24};

    @Builder.Default
    private int visionHiddenSize = 1152;

    @Builder.Default
    private int llmHiddenSize = 4096;

    /**
     * Merge intermediate ViT features into LLM hidden states.
     *
     * <p>Each ViT feature is projected from visionHiddenSize to llmHiddenSize
     * and added to the LLM hidden state via a residual connection.</p>
     *
     * @param llmHidden the LLM hidden states [batch, seq, llmHidden]
     * @param vitFeatures map of ViT layer index -> features [batch, visionSeq, visionHidden]
     * @param projectionWeights map of layer index -> projection weight [visionHidden, llmHidden]
     * @return merged hidden states [batch, seq, llmHidden]
     */
    public INDArray merge(INDArray llmHidden,
                          Map<Integer, INDArray> vitFeatures,
                          Map<Integer, INDArray> projectionWeights) {
        INDArray result = llmHidden.dup();

        for (int layerIdx : visualIndexes) {
            INDArray vitFeature = vitFeatures.get(layerIdx);
            INDArray projWeight = projectionWeights.get(layerIdx);

            if (vitFeature == null) {
                log.warn("Missing ViT feature for layer {}, skipping", layerIdx);
                continue;
            }
            if (projWeight == null) {
                log.warn("Missing projection weight for layer {}, skipping", layerIdx);
                continue;
            }

            // Project vision features: [batch, visionSeq, visionHidden] x [visionHidden, llmHidden]
            // -> [batch, visionSeq, llmHidden]
            INDArray projected = vitFeature.mmul(projWeight);

            // Add to LLM hidden states via residual
            // Vision tokens replace specific positions in the sequence
            long visionSeqLen = projected.size(1);
            long llmSeqLen = result.size(1);

            if (visionSeqLen <= llmSeqLen) {
                // Add projected vision features to the first visionSeqLen positions
                INDArray target = result.get(
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.interval(0, visionSeqLen),
                        org.nd4j.linalg.indexing.NDArrayIndex.all()
                );
                target.addi(projected);
            } else {
                log.warn("Vision seq ({}) > LLM seq ({}), truncating", visionSeqLen, llmSeqLen);
                INDArray truncated = projected.get(
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.interval(0, llmSeqLen),
                        org.nd4j.linalg.indexing.NDArrayIndex.all()
                );
                result.addi(truncated);
            }

            projected.close();
            log.debug("Merged ViT layer {} features: vision_seq={}, projected to llm_hidden={}",
                    layerIdx, visionSeqLen, llmHiddenSize);
        }

        return result;
    }

    /**
     * Build DeepStack merge operations into a SameDiff graph.
     *
     * @param sd the SameDiff instance
     * @param llmHidden the LLM hidden state variable
     * @param vitFeatures map of ViT layer index -> feature variable
     * @param namePrefix prefix for variable names
     * @return the merged hidden state variable
     */
    public SDVariable mergeGraph(SameDiff sd, SDVariable llmHidden,
                                  Map<Integer, SDVariable> vitFeatures,
                                  String namePrefix) {
        SDVariable result = llmHidden;

        for (int layerIdx : visualIndexes) {
            SDVariable vitFeature = vitFeatures.get(layerIdx);
            if (vitFeature == null) continue;

            String projName = namePrefix + "_deepstack_proj_" + layerIdx;
            SDVariable projWeight = sd.var(projName + "_weight",
                    org.nd4j.linalg.api.buffer.DataType.FLOAT,
                    visionHiddenSize, llmHiddenSize);

            // Project: [batch, visionSeq, visionHidden] x [visionHidden, llmHidden]
            SDVariable projected = sd.mmul(vitFeature, projWeight);

            // Residual add
            result = sd.math.add(namePrefix + "_deepstack_merged_" + layerIdx,
                    result, projected);
        }

        return result;
    }

    /**
     * Get the number of DeepStack levels.
     */
    public int getNumLevels() {
        return visualIndexes.length;
    }

    /**
     * Create a DeepStack merger for Qwen3-VL.
     * Extracts features at ViT layers [8, 16, 24] with vision_hidden=1152, llm_hidden=4096.
     */
    public static DeepStackMerger forQwen3VL() {
        return DeepStackMerger.builder()
                .visualIndexes(new int[]{8, 16, 24})
                .visionHiddenSize(1152)
                .llmHiddenSize(4096)
                .build();
    }

    /**
     * Create a DeepStack merger for Qwen3-VL-2B (smaller model).
     */
    public static DeepStackMerger forQwen3VL2B() {
        return DeepStackMerger.builder()
                .visualIndexes(new int[]{8, 16, 24})
                .visionHiddenSize(1152)
                .llmHiddenSize(1536)
                .build();
    }
}
