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

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Merges text and vision embeddings by replacing &lt;image&gt; token positions
 * with vision embeddings.
 *
 * Uses host-side float[] copy for CUDA safety - Nd4j.concat() and .put()
 * have known CUDA sync issues where the device buffer doesn't match the host buffer.
 */
@Slf4j
public class EmbeddingMerger {

    /**
     * Replace &lt;image&gt; token embeddings with vision embeddings (ONNX reference behavior).
     *
     * Builds the merged embedding array via host-side float[] to guarantee proper CUDA
     * host-&gt;device sync.
     *
     * @param textEmbeddings text embeddings [1, seqLen, hiddenDim]
     * @param visionEmbeddings vision embeddings [1, visionSeqLen, hiddenDim]
     * @param tokenIds the prompt token IDs
     * @param imageTokenId the &lt;image&gt; token ID
     * @return merged embeddings [1, seqLen, hiddenDim] with vision tokens replacing image positions
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
        if (visionFlat.dataType() != textEmbeddings.dataType()) {
            log.info("Casting vision embeddings from {} to {} to match text embeddings",
                    visionFlat.dataType(), textEmbeddings.dataType());
            visionFlat = visionFlat.castTo(textEmbeddings.dataType());
        }

        int seqLen = tokenIds.length;
        int hiddenDim = (int) visionHiddenSize;

        // Extract raw float data from both embedding sources via host buffers
        float[] textData = textEmbeddings.data().asFloat();
        float[] visionData = visionFlat.data().asFloat();

        // Build merged float array on the host
        float[] mergedData = new float[seqLen * hiddenDim];
        int fillIdx = 0;
        int fillCount = Math.min(imageSlots, (int) visionSeqLen);
        for (int pos = 0; pos < seqLen; pos++) {
            int destOffset = pos * hiddenDim;
            if (tokenIds[pos] == imageTokenId && fillIdx < fillCount) {
                int srcOffset = fillIdx * hiddenDim;
                System.arraycopy(visionData, srcOffset, mergedData, destOffset, hiddenDim);
                if (fillIdx == 0) {
                    float vMin = Float.MAX_VALUE, vMax = -Float.MAX_VALUE;
                    for (int k = 0; k < hiddenDim; k++) {
                        vMin = Math.min(vMin, mergedData[destOffset + k]);
                        vMax = Math.max(vMax, mergedData[destOffset + k]);
                    }
                    log.info("Vision slice [0] host data: min={}, max={}", vMin, vMax);
                }
                fillIdx++;
            } else {
                int srcOffset = pos * hiddenDim;
                System.arraycopy(textData, srcOffset, mergedData, destOffset, hiddenDim);
            }
        }

        // Create fresh INDArray from host float[] - guarantees proper CUDA device sync
        INDArray inputsEmbeds = Nd4j.create(mergedData, new long[]{1, seqLen, hiddenDim}, 'c');
        log.info("Built inputsEmbeds via host float[]: shape={}", java.util.Arrays.toString(inputsEmbeds.shape()));
        log.info("Filled {} of {} image token positions", fillIdx, imageSlots);
        if (fillIdx < (int) visionSeqLen) {
            log.warn("Only filled {} of {} vision tokens", fillIdx, visionSeqLen);
        }

        return inputsEmbeds;
    }
}
