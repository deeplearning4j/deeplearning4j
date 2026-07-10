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

import java.util.Arrays;
import java.util.List;

/**
 * KV Cache Quantize operation.
 *
 * <p>Quantizes floating-point KV cache data using per-row symmetric quantization.
 * Supports INT8, FP8_E4M3, FP8_E5M2, and INT4 formats.</p>
 *
 * <p>Quantization formats:
 * <ul>
 *   <li>0 = INT8: scale = max(abs(row)) / 127</li>
 *   <li>1 = FP8_E4M3: FP8 with 4-bit exponent, 3-bit mantissa</li>
 *   <li>2 = FP8_E5M2: FP8 with 5-bit exponent, 2-bit mantissa</li>
 *   <li>3 = INT4: scale = max(abs(row)) / 7, packed 2 per byte</li>
 * </ul>
 * </p>
 *
 * <p>Produces two outputs:
 * <ol>
 *   <li>quantized tensor (INT8 dtype)</li>
 *   <li>scales tensor (FLOAT32 dtype, one scale per row)</li>
 * </ol>
 * </p>
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class KVCacheQuantize extends DynamicCustomOp {

    /** INT8 symmetric quantization */
    public static final int FORMAT_INT8 = 0;
    /** FP8 E4M3 format */
    public static final int FORMAT_FP8_E4M3 = 1;
    /** FP8 E5M2 format */
    public static final int FORMAT_FP8_E5M2 = 2;
    /** INT4 packed quantization */
    public static final int FORMAT_INT4 = 3;

    /**
     * Create a KV cache quantize op for INDArray execution.
     *
     * @param input       float KV data to quantize
     * @param quantFormat quantization format (0=INT8, 1=FP8_E4M3, 2=FP8_E5M2, 3=INT4)
     */
    public KVCacheQuantize(INDArray input, int quantFormat) {
        super(new INDArray[]{input}, null);
        addIArgument(quantFormat);
    }

    /**
     * ADR 0107 V2 ROW-INLINE (INT8 only). When {@code inline} is true, output 0 is a row-inline
     * INT8 tensor of the input shape with the last dimension extended by 4: each row holds
     * {@code rowLen} int8 values followed by that row's float32 scale. The scales live INSIDE the
     * logical tensor, so DSP ext-input staging, H2D re-staging and CUDA-graph capture preserve
     * them. Output 1 is an unused dummy scalar. Feed output 0 directly as the INT8 keyCache /
     * valueCache (read by fusedGQADecodeQuantisedCuda / the Triton flash_attn INT8 path).
     */
    public KVCacheQuantize(INDArray input, int quantFormat, boolean inline) {
        super(new INDArray[]{input}, null);
        addIArgument(quantFormat, inline ? 1 : 0);
    }

    /**
     * Create a KV cache quantize op for SameDiff graph execution.
     *
     * @param sd          SameDiff instance
     * @param input       input variable
     * @param quantFormat quantization format
     */
    public KVCacheQuantize(SameDiff sd, SDVariable input, int quantFormat) {
        super(null, sd, new SDVariable[]{input}, false);
        addIArgument(quantFormat);
    }

    public KVCacheQuantize(SameDiff sd, SDVariable input, int quantType, int groupSize) {
        super(null, sd, new SDVariable[]{input}, false);
        addIArgument(quantType, groupSize);
    }

    public KVCacheQuantize(INDArray input, int quantType, int groupSize) {
        super(new INDArray[]{input}, null);
        addIArgument(quantType, groupSize);
    }

    @Override
    public String opName() {
        return "kv_cache_quantize";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Arrays.asList(DataType.INT8, DataType.FLOAT);
    }

    @Override
    public int getNumOutputs() {
        return 2;
    }
}
