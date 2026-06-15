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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

/**
 * 3D-Resampler for MiniCPM-V 4.5 video token compression.
 *
 * <p>Groups N consecutive video frames (default 6) and compresses them into K tokens
 * (default 64) using cross-attention with learnable query tokens. This achieves
 * a 96x compression rate for video tokens compared to per-frame encoding.</p>
 *
 * <p>The compression pipeline:</p>
 * <ol>
 *   <li>Group N consecutive frame embeddings: [batch, N, tokens_per_frame, hidden]</li>
 *   <li>Flatten to key/value: [batch, N * tokens_per_frame, hidden]</li>
 *   <li>Cross-attend with K learnable queries: [batch, K, hidden]</li>
 *   <li>Output: [batch, K, hidden] (K compressed tokens per group)</li>
 * </ol>
 *
 * <p>For a 30s video at 10 FPS = 300 frames:</p>
 * <ul>
 *   <li>Per-frame: 300 * 576 = 172,800 tokens</li>
 *   <li>After 3D-Resampler: (300/6) * 64 = 3,200 tokens (54x compression)</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * ThreeDResampler resampler = ThreeDResampler.builder()
 *     .frameGroupSize(6).outputTokensPerGroup(64).hiddenSize(1152).numHeads(16).build();
 *
 * // frameEmbeddings: list of [1, tokens, hidden] per frame
 * INDArray compressed = resampler.resample(frameEmbeddings, queryWeights, kWeight, vWeight, oWeight);
 * // compressed: [1, numGroups * 64, hidden]
 * }</pre>
 *
 * @see VideoVisionLanguageModel
 */
@Slf4j
@Builder
@Getter
public class ThreeDResampler {

    @Builder.Default
    private int frameGroupSize = 6;

    @Builder.Default
    private int outputTokensPerGroup = 64;

    @Builder.Default
    private int hiddenSize = 1152;

    @Builder.Default
    private int numHeads = 16;

    /**
     * Resample frame embeddings using cross-attention compression.
     *
     * @param frameEmbeddings list of per-frame embeddings, each [1, tokens, hidden]
     * @param queries learnable query tokens [outputTokensPerGroup, hidden]
     * @param keyWeight key projection [hidden, hidden]
     * @param valueWeight value projection [hidden, hidden]
     * @param outputWeight output projection [hidden, hidden]
     * @return compressed tokens [1, totalOutputTokens, hidden]
     */
    public INDArray resample(List<INDArray> frameEmbeddings,
                             INDArray queries,
                             INDArray keyWeight,
                             INDArray valueWeight,
                             INDArray outputWeight) {
        int numFrames = frameEmbeddings.size();
        int numGroups = (numFrames + frameGroupSize - 1) / frameGroupSize;

        List<INDArray> groupOutputs = new ArrayList<>(numGroups);

        for (int g = 0; g < numGroups; g++) {
            int startFrame = g * frameGroupSize;
            int endFrame = Math.min(startFrame + frameGroupSize, numFrames);

            // Collect frames in this group
            List<INDArray> groupFrames = new ArrayList<>();
            for (int f = startFrame; f < endFrame; f++) {
                INDArray frame = frameEmbeddings.get(f);
                // Remove batch dim if present: [1, tokens, hidden] -> [tokens, hidden]
                if (frame.rank() == 3) {
                    frame = frame.reshape(frame.size(1), frame.size(2));
                }
                groupFrames.add(frame);
            }

            // Concatenate all frames in group: [totalTokens, hidden]
            INDArray groupTokens;
            if (groupFrames.size() == 1) {
                groupTokens = groupFrames.get(0);
            } else {
                groupTokens = Nd4j.concat(0, groupFrames.toArray(new INDArray[0]));
            }

            // Project keys and values: [totalTokens, hidden]
            INDArray keys = groupTokens.mmul(keyWeight);
            INDArray values = groupTokens.mmul(valueWeight);

            // Cross-attention: queries attend to keys/values
            // Q: [outputTokens, hidden], K: [totalTokens, hidden], V: [totalTokens, hidden]
            int headDim = hiddenSize / numHeads;
            float scale = (float) (1.0 / Math.sqrt(headDim));

            // Simple scaled dot-product attention
            // scores: [outputTokens, totalTokens]
            INDArray scores = queries.mmul(keys.transpose()).mul(scale);

            // Softmax over last dimension
            INDArray maxScores = scores.max(true, 1);
            INDArray expScores = scores.sub(maxScores);
            Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.Exp(expScores, expScores));
            INDArray sumExp = expScores.sum(true, 1);
            INDArray attnWeights = expScores.div(sumExp);

            // Weighted sum: [outputTokens, hidden]
            INDArray attended = attnWeights.mmul(values);

            // Output projection
            INDArray output = attended.mmul(outputWeight);

            groupOutputs.add(output);

            // Clean up intermediates
            keys.close();
            values.close();
            scores.close();
            maxScores.close();
            expScores.close();
            sumExp.close();
            attnWeights.close();
            attended.close();

            if (groupFrames.size() > 1) {
                groupTokens.close();
            }
        }

        // Concatenate all group outputs: [totalOutputTokens, hidden]
        INDArray allOutputs;
        if (groupOutputs.size() == 1) {
            allOutputs = groupOutputs.get(0);
        } else {
            allOutputs = Nd4j.concat(0, groupOutputs.toArray(new INDArray[0]));
            for (INDArray go : groupOutputs) {
                go.close();
            }
        }

        // Add batch dimension: [1, totalOutputTokens, hidden]
        INDArray result = allOutputs.reshape(1, allOutputs.size(0), allOutputs.size(1));

        int totalOutputTokens = numGroups * outputTokensPerGroup;
        log.info("3D-Resampler: {} frames -> {} groups -> {} tokens ({}x compression)",
                numFrames, numGroups, result.size(1),
                numFrames > 0 ? (numFrames * frameEmbeddings.get(0).size(1)) / result.size(1) : 0);

        return result;
    }

    /**
     * Build the 3D-Resampler into a SameDiff graph.
     *
     * @param sd the SameDiff instance
     * @param frameEmbeddings concatenated frame embeddings [batch, totalFrameTokens, hidden]
     * @param numFrames number of frames in the input
     * @param namePrefix variable name prefix
     * @return compressed output [batch, numGroups * outputTokens, hidden]
     */
    public SDVariable buildGraph(SameDiff sd, SDVariable frameEmbeddings,
                                  int numFrames, String namePrefix) {
        // Learnable query tokens
        SDVariable queries = sd.var(namePrefix + "_queries",
                DataType.FLOAT, 1, outputTokensPerGroup, hiddenSize);

        // Projection weights
        SDVariable kWeight = sd.var(namePrefix + "_k_proj", DataType.FLOAT, hiddenSize, hiddenSize);
        SDVariable vWeight = sd.var(namePrefix + "_v_proj", DataType.FLOAT, hiddenSize, hiddenSize);
        SDVariable oWeight = sd.var(namePrefix + "_o_proj", DataType.FLOAT, hiddenSize, hiddenSize);

        // Project keys and values
        SDVariable keys = sd.mmul(frameEmbeddings, kWeight);
        SDVariable values = sd.mmul(frameEmbeddings, vWeight);

        // Cross-attention using DotProductAttentionV2
        SDVariable attended = sd.nn().dotProductAttention(
                namePrefix + "_cross_attn",
                queries, keys, values, null, true);

        // Output projection
        return sd.mmul(namePrefix + "_output", attended, oWeight);
    }

    /**
     * Get number of output tokens for a given number of input frames.
     */
    public int getOutputTokenCount(int numFrames) {
        int numGroups = (numFrames + frameGroupSize - 1) / frameGroupSize;
        return numGroups * outputTokensPerGroup;
    }

    /**
     * Get the compression ratio for a given number of frames and tokens per frame.
     */
    public double getCompressionRatio(int numFrames, int tokensPerFrame) {
        int inputTokens = numFrames * tokensPerFrame;
        int outputTokens = getOutputTokenCount(numFrames);
        return (double) inputTokens / outputTokens;
    }

}
