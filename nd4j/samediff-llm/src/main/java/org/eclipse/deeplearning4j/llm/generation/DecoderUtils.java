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

package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Utilities for decoder model execution: causal masks, KV cache, and output name resolution.
 */
@Slf4j
public class DecoderUtils {

    /** Mask fill value matching torch.finfo(torch.float32).min */
    public static final float MASK_FILL = -3.4028235e+38f;

    public static class KVCacheNames {
        public final List<String> keyNames;
        public final List<String> valueNames;

        public KVCacheNames(List<String> keyNames, List<String> valueNames) {
            this.keyNames = keyNames;
            this.valueNames = valueNames;
        }
    }

    /**
     * Build a causal attention mask for the decoder (FLOAT dtype).
     *
     * For prefill (currentSeqLen &gt; 1): upper-triangular mask filled with MASK_FILL
     *   mask[q][k] = 0 if k &lt;= (pastSeqLen + q), else MASK_FILL
     *
     * For decode (currentSeqLen == 1): all zeros (single token attends to all past)
     *
     * Uses FLOAT (not LONG) because AddOp doesn't promote LONG-&gt;FLOAT correctly.
     *
     * @param currentSeqLen number of query tokens in this step
     * @param totalSeqLen total KV length (past + current)
     * @return INDArray of shape [1, 1, currentSeqLen, totalSeqLen] with FLOAT dtype
     */
    public static INDArray buildCausalMask(long currentSeqLen, long totalSeqLen) {
        if (currentSeqLen == 1) {
            return Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeqLen);
        }

        long pastSeqLen = totalSeqLen - currentSeqLen;
        int Q = (int) currentSeqLen;
        int K = (int) totalSeqLen;
        float[] data = new float[Q * K];

        for (int q = 0; q < Q; q++) {
            int lastVisibleK = (int) pastSeqLen + q;
            int rowOffset = q * K;
            for (int k = lastVisibleK + 1; k < K; k++) {
                data[rowOffset + k] = MASK_FILL;
            }
        }

        INDArray mask = Nd4j.createFromArray(data).reshape(1, 1, currentSeqLen, totalSeqLen);
        long nonZero = mask.neq(0).castTo(DataType.LONG).sumNumber().longValue();
        log.info("Built causal mask: shape=[1, 1, {}, {}], pastSeqLen={}, non-zero count={}, dtype=FLOAT",
                currentSeqLen, totalSeqLen, pastSeqLen, nonZero);
        return mask;
    }

    /**
     * Create an empty KV cache tensor for the first decoder step.
     * Shape is [batchSize, numHeads, 0, headDim].
     *
     * Infers numHeads and headDim from the decoder graph's variable shapes.
     *
     * @param decoder the SameDiff decoder model
     * @param inputName the past_key_values input name
     * @param batchSize batch size
     * @param hiddenSize model hidden size (used for fallback inference)
     * @return empty KV cache INDArray
     */
    public static INDArray createEmptyKvCache(SameDiff decoder, String inputName, long batchSize, long hiddenSize) {
        long numHeads = -1;
        long headDim = -1;
        DataType kvType = DataType.FLOAT;

        SDVariable inputVar = decoder.getVariable(inputName);
        if (inputVar != null && inputVar.getShape() != null && inputVar.getShape().length >= 4) {
            long[] shape = inputVar.getShape();
            if (inputVar.dataType() != null) {
                kvType = inputVar.dataType();
            }
            if (shape[1] > 0) {
                numHeads = shape[1];
            }
            if (shape[3] > 0) {
                headDim = shape[3];
            }
        }

        if (numHeads <= 0 || headDim <= 0) {
            String presentName = inputName.replace("past_key_values", "present");
            SDVariable presentVar = decoder.getVariable(presentName);
            if (presentVar != null && presentVar.getShape() != null && presentVar.getShape().length >= 4) {
                long[] shape = presentVar.getShape();
                if (numHeads <= 0 && shape[1] > 0) {
                    numHeads = shape[1];
                }
                if (headDim <= 0 && shape[3] > 0) {
                    headDim = shape[3];
                }
            }
        }

        if (headDim <= 0 && numHeads > 0 && hiddenSize > 0) {
            headDim = Math.max(1, hiddenSize / numHeads);
        }
        if (numHeads <= 0 && headDim > 0 && hiddenSize > 0) {
            numHeads = Math.max(1, hiddenSize / headDim);
        }
        if (headDim <= 0) {
            headDim = 64;
        }
        if (numHeads <= 0) {
            numHeads = Math.max(1, hiddenSize / headDim);
        }

        return Nd4j.zeros(kvType, batchSize, numHeads, 0, headDim);
    }

    /**
     * Find the logits output variable name from a decoder model.
     *
     * @param decoder the SameDiff decoder model
     * @return the logits output name, or null if not found
     */
    public static String findLogitsOutputName(SameDiff decoder) {
        for (String outputName : decoder.outputs()) {
            if (outputName.contains("logit") || outputName.equals("logits")) {
                return outputName;
            }
        }
        if (!decoder.outputs().isEmpty()) {
            return decoder.outputs().get(0);
        }
        return null;
    }

    /**
     * Find the KV cache output names (present key/value) from a decoder model.
     *
     * @param decoder the SameDiff decoder model
     * @return KVCacheNames with sorted key and value output names
     */
    public static KVCacheNames findKVCacheOutputNames(SameDiff decoder) {
        List<String> presentKeyNames = new ArrayList<>();
        List<String> presentValueNames = new ArrayList<>();

        for (String outputName : decoder.outputs()) {
            if (outputName.contains("present") && outputName.contains("key")) {
                presentKeyNames.add(outputName);
            } else if (outputName.contains("present") && outputName.contains("value")) {
                presentValueNames.add(outputName);
            }
        }

        Collections.sort(presentKeyNames);
        Collections.sort(presentValueNames);

        return new KVCacheNames(presentKeyNames, presentValueNames);
    }
}
