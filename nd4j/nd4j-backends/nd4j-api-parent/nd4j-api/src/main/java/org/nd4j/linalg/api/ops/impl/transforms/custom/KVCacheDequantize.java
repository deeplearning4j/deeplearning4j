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
 * KV Cache Dequantize operation.
 *
 * <p>Reverses {@link KVCacheQuantize}: reconstructs floating-point KV data
 * from quantized representation and per-row scales.</p>
 *
 * <p>output = quantized * scale (per row)</p>
 *
 * <p>Quantization formats:
 * <ul>
 *   <li>0 = INT8</li>
 *   <li>1 = FP8_E4M3</li>
 *   <li>2 = FP8_E5M2</li>
 *   <li>3 = INT4</li>
 * </ul>
 * </p>
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class KVCacheDequantize extends DynamicCustomOp {

    /**
     * Create a KV cache dequantize op for INDArray execution.
     *
     * @param quantized   quantized INT8 data
     * @param scales      per-row scales from quantization
     * @param quantFormat quantization format (0=INT8, 1=FP8_E4M3, 2=FP8_E5M2, 3=INT4)
     */
    public KVCacheDequantize(INDArray quantized, INDArray scales, int quantFormat) {
        super(new INDArray[]{quantized, scales}, null);
        addIArgument(quantFormat);
    }

    /**
     * Create a KV cache dequantize op for SameDiff graph execution.
     *
     * @param sd          SameDiff instance
     * @param quantized   quantized variable
     * @param scales      scales variable
     * @param quantFormat quantization format
     */
    public KVCacheDequantize(SameDiff sd, SDVariable quantized, SDVariable scales, int quantFormat) {
        super(null, sd, new SDVariable[]{quantized, scales}, false);
        addIArgument(quantFormat);
    }

    @Override
    public String opName() {
        return "kv_cache_dequantize";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(DataType.FLOAT);
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
