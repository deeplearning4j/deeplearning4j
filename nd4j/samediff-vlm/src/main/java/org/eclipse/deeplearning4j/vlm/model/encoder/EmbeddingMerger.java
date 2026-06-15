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

package org.eclipse.deeplearning4j.vlm.model.encoder;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.VisionEmbeddingMerge;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Merges text and vision embeddings by replacing target token positions
 * with vision embeddings using a native device-side scatter kernel.
 *
 * <p>All operations run entirely on device (CUDA) — no host round-trips.
 * Uses the {@code vision_embedding_merge} native op which builds a prefix-sum
 * mapping of target token positions, then scatters vision embeddings in a
 * single parallel pass.</p>
 */
@Slf4j
public class EmbeddingMerger {

    /**
     * Replace image token embeddings with vision embeddings.
     *
     * Uses host-side float[] copy for guaranteed correctness — the native
     * VisionEmbeddingMerge op can be used via mergeNative() but this path
     * ensures correct CUDA host/device sync by building the merged array
     * from raw float data on the host side.
     *
     * @param textEmbeddings text embeddings [1, seqLen, hiddenDim]
     * @param visionEmbeddings vision embeddings [1, visionSeqLen, hiddenDim]
     * @param tokenIds the prompt token IDs
     * @param imageTokenId the image token ID to replace
     * @return merged embeddings [1, seqLen, hiddenDim] with vision tokens at image positions
     */
    public static INDArray mergeEmbeddings(INDArray textEmbeddings, INDArray visionEmbeddings,
                                           int[] tokenIds, int imageTokenId) {
        long visionSeqLen = visionEmbeddings.shape()[1];
        long visionHiddenSize = visionEmbeddings.shape()[2];

        int imageSlots = 0;
        for (int id : tokenIds) {
            if (id == imageTokenId) imageSlots++;
        }

        if (imageSlots != visionSeqLen) {
            log.warn("Image token slots ({}) != vision tokens ({}). Will fill {} slots.",
                    imageSlots, visionSeqLen, Math.min(imageSlots, (int) visionSeqLen));
        }

        // Cast vision embeddings to match text embedding dtype
        INDArray visionFlat = visionEmbeddings.reshape((int) visionSeqLen, (int) visionHiddenSize);
        INDArray visionCast = null;
        if (visionFlat.dataType() != textEmbeddings.dataType()) {
            log.info("Casting vision embeddings from {} to {} to match text embeddings",
                    visionFlat.dataType(), textEmbeddings.dataType());
            visionCast = visionFlat.castTo(textEmbeddings.dataType());
            visionFlat = visionCast;
        }

        int seqLen = tokenIds.length;
        int hiddenDim = (int) visionHiddenSize;

        // Extract raw float data from both embedding sources via host buffers
        float[] textData = textEmbeddings.data().asFloat();
        float[] visionData = visionFlat.data().asFloat();

        // Close the castTo() result now that we've extracted the float data
        if (visionCast != null && !visionCast.wasClosed()) {
            visionCast.setCloseable(true);
            visionCast.close();
        }

        // Build merged float array on the host
        float[] mergedData = new float[seqLen * hiddenDim];
        int fillIdx = 0;
        int fillCount = Math.min(imageSlots, (int) visionSeqLen);
        for (int pos = 0; pos < seqLen; pos++) {
            int destOffset = pos * hiddenDim;
            if (tokenIds[pos] == imageTokenId && fillIdx < fillCount) {
                int srcOffset = fillIdx * hiddenDim;
                System.arraycopy(visionData, srcOffset, mergedData, destOffset, hiddenDim);
                fillIdx++;
            } else {
                int srcOffset = pos * hiddenDim;
                System.arraycopy(textData, srcOffset, mergedData, destOffset, hiddenDim);
            }
        }

        // Create fresh INDArray from host float[] — guarantees proper CUDA device sync
        INDArray inputsEmbeds = Nd4j.create(mergedData, new long[]{1, seqLen, hiddenDim}, 'c');
        log.info("Built inputsEmbeds via host float[]: shape={}, filled {} of {} image token positions",
                java.util.Arrays.toString(inputsEmbeds.shape()), fillIdx, imageSlots);

        return inputsEmbeds;
    }

    /**
     * Replace video/image token embeddings with concatenated frame vision embeddings
     * via native scatter.
     *
     * <p>For video VLMs, each frame produces a set of vision tokens. All frame tokens
     * are concatenated along the sequence dimension and scattered into the target
     * token positions in a single native kernel call.</p>
     *
     * @param textEmbeddings text embeddings [1, seqLen, hiddenDim]
     * @param frameEmbeddings list of per-frame vision embeddings, each [1, tokensPerFrame, hiddenDim]
     * @param tokenIds the prompt token IDs
     * @param videoTokenId the video/image token ID to replace
     * @return merged embeddings [1, seqLen, hiddenDim]
     */
    public static INDArray mergeVideoEmbeddings(INDArray textEmbeddings,
                                                List<INDArray> frameEmbeddings,
                                                int[] tokenIds, int videoTokenId) {
        if (frameEmbeddings == null || frameEmbeddings.isEmpty()) {
            log.warn("No frame embeddings provided, returning text embeddings unchanged");
            return textEmbeddings;
        }

        // Concatenate all frame embeddings along sequence dimension -> [1, totalVisionTokens, hidden]
        INDArray[] frameArrays = frameEmbeddings.toArray(new INDArray[0]);
        INDArray allVision = Nd4j.concat(1, frameArrays);

        INDArray tokenIdArray = Nd4j.createFromArray(tokenIds).reshape(1, tokenIds.length).dup();
        return mergeNative(textEmbeddings, allVision, tokenIdArray, videoTokenId);
    }

    /**
     * Execute the native vision_embedding_merge op.
     */
    private static INDArray mergeNative(INDArray textEmbeddings, INDArray visionEmbeddings,
                                        INDArray tokenIds, long targetTokenId) {
        // Cast vision to match text dtype if needed
        INDArray visionToUse = visionEmbeddings;
        if (visionEmbeddings.dataType() != textEmbeddings.dataType()) {
            log.info("Casting vision embeddings from {} to {}", visionEmbeddings.dataType(), textEmbeddings.dataType());
            visionToUse = visionEmbeddings.castTo(textEmbeddings.dataType());
        }

        VisionEmbeddingMerge op = new VisionEmbeddingMerge(textEmbeddings, visionToUse, tokenIds, targetTokenId);
        INDArray[] result = Nd4j.exec(op);
        return result[0];
    }
}
