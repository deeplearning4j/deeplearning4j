/*
 *  ******************************************************************************
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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Decoder inputs for tests that drive an imported decoder directly through the in-graph KV
 * contract (ONNX MHA over fixed caches the graph writes at {@code cache_position}), built the
 * way generateNative builds them. Tests that assembled the older external-KV inputs themselves
 * failed with "missing external input 'causal_mask'" once the decoder moved to this contract.
 */
public final class InGraphKvDecoderInputs {

    private InGraphKvDecoderInputs() {
    }

    /** Zeroed caches in the decoder's fixed [batch, heads, maxKvLen, headDim] layout. */
    public static Map<String, INDArray> kvBuffers(SameDiff decoder, long maxKvLen) {
        ModelIOConfig.KVCacheNames kvNames = ModelIOConfig.findKVCacheInputNames(decoder);
        List<String> names = new ArrayList<>(kvNames.keyNames);
        names.addAll(kvNames.valueNames);
        Map<String, INDArray> buffers = new LinkedHashMap<>();
        for (String name : names) {
            SDVariable placeholder = decoder.getVariable(name);
            long[] declared = placeholder.getShape();
            buffers.put(name, Nd4j.zeros(placeholder.dataType(),
                    declared[0] > 0 ? declared[0] : 1, declared[1], maxKvLen, declared[3]));
        }
        return buffers;
    }

    /**
     * Decoder inputs for one step: DecoderInputBuilder fills the step inputs over fixed caches
     * that the graph writes at cache_position, and causal_mask is a bias over the caches'
     * maxKvLen positions, where the builder's own mask has the external-concat width. The caller
     * owns the returned arrays except the caches and {@code embeddings}, which it passed in.
     */
    public static Map<String, INDArray> stepInputs(SameDiff decoder, long hiddenSize, INDArray embeddings,
                                                   long cachePos, Map<String, INDArray> kvBuffers) {
        assertTrue(ModelIOConfig.isOnnxMhaInPlaceKvCache(decoder),
                "these inputs follow the in-graph ONNX MHA cache contract");
        ModelIOConfig io = ModelIOConfig.discover(decoder);
        long seqLen = embeddings.size(1);
        long maxKvLen = kvBuffers.values().iterator().next().size(2);
        INDArray inputIds = Nd4j.zeros(DataType.INT64, 1, seqLen);
        Map<String, INDArray> inputs = DecoderInputBuilder.buildDecoderInputMap(io, decoder.inputs(),
                decoder, embeddings, inputIds, cachePos, seqLen, kvBuffers, maxKvLen, cachePos,
                true, hiddenSize, null, true, null, null, seqLen);
        boolean idsUsed = false;
        for (INDArray value : inputs.values()) {
            idsUsed |= value == inputIds;
        }
        if (!idsUsed) {
            inputIds.close();
        }
        String causalName = io.getCausalMaskName();
        DataType maskType = decoder.getVariable(causalName).dataType();
        INDArray mask = cachePos == 0
                ? DecoderInputBuilder.buildInGraphCausalMask(seqLen, maxKvLen, maskType)
                : decodeCausalMask(cachePos, maxKvLen, maskType);
        INDArray builderMask = inputs.put(causalName, mask);
        if (builderMask != null) {
            builderMask.close();
        }
        return inputs;
    }

    /** One decode row at cachePos: the cached positions and the new token are visible. */
    public static INDArray decodeCausalMask(long cachePos, long maxKvLen, DataType dtype) {
        float[] row = new float[(int) maxKvLen];
        for (int k = (int) cachePos + 1; k < maxKvLen; k++) {
            row[k] = ModelIOConfig.MASK_FILL;
        }
        INDArray mask = Nd4j.createFromArray(row).reshape(1, 1, 1, maxKvLen);
        if (dtype == DataType.FLOAT) {
            return mask;
        }
        INDArray cast = mask.castTo(dtype);
        mask.close();
        return cast;
    }
}
