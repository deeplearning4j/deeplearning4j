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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * KV cache scatter update for LLM autoregressive decoding.
 *
 * Copies a single time-step slice from present KV tensor into the
 * static KV buffer at a given cache position. Uses inplace operation
 * so the framework properly tracks device buffer modifications.
 *
 * Input 0: static_kv  [batch, heads, maxKvLen, dim] (destination, modified in-place)
 * Input 1: present_kv [batch, heads, seqLen, dim] (source, last position extracted)
 *
 * Int args:
 *   0: cachePos — position in static buffer to write to
 *
 * Output 0: same as input 0 (inplace)
 */
@NoArgsConstructor
public class KvScatter extends DynamicCustomOp {

    /**
     * INDArray constructor.
     *
     * @param staticBuffer static KV cache buffer (updated in-place, returned as output)
     * @param present      present KV tensor from decoder output
     * @param cachePos     position in static buffer to write the new entry
     */
    public KvScatter(INDArray staticBuffer, INDArray present, long cachePos) {
        super(null, new INDArray[]{staticBuffer, present}, null);
        addIArgument(cachePos);
    }

    /**
     * INDArray constructor (codegen-compatible with numPairs parameter).
     * Note: present is input 1, staticBuffer is input 0 in the C++ op.
     */
    public KvScatter(INDArray present, INDArray staticBuffer, long cachePos, int numPairs) {
        super(null, new INDArray[]{staticBuffer, present}, null);
        addIArgument(cachePos);
    }

    /**
     * Batched INDArray constructor. Scatters multiple KV pairs in a single kernel launch.
     * Arrays are interleaved: static0, present0, static1, present1, ...
     *
     * @param staticBuffers array of static KV cache buffers [batch, heads, maxKvLen, dim]
     * @param presents      array of present KV tensors [batch, heads, seqLen, dim]
     * @param cachePos      position in static buffer to write to
     */
    public KvScatter(INDArray[] staticBuffers, INDArray[] presents, long cachePos) {
        super(null, interleaveArrays(staticBuffers, presents), null);
        addIArgument(cachePos);
    }

    private static INDArray[] interleaveArrays(INDArray[] a, INDArray[] b) {
        INDArray[] result = new INDArray[a.length + b.length];
        for (int i = 0; i < a.length; i++) {
            result[2 * i] = a[i];
            result[2 * i + 1] = b[i];
        }
        return result;
    }

    /**
     * SameDiff constructor.
     */
    public KvScatter(SameDiff sameDiff, SDVariable staticBuffer, SDVariable present,
                     long cachePos) {
        super(null, sameDiff, new SDVariable[]{staticBuffer, present}, true);
        addIArgument(cachePos);
    }

    /**
     * SameDiff constructor (codegen-compatible with numPairs parameter).
     * Note: present is input 1, staticBuffer is input 0 in the C++ op.
     */
    public KvScatter(SameDiff sameDiff, SDVariable present, SDVariable staticBuffer,
                     long cachePos, int numPairs) {
        super(null, sameDiff, new SDVariable[]{staticBuffer, present}, true);
        addIArgument(cachePos);
    }

    @Override
    public String opName() {
        return "kv_scatter";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
