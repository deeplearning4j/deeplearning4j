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

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * FP8 Matrix Multiplication operation (SM 89+ / Ada Lovelace and newer).
 * <p>
 * Performs GEMM with FP8 quantized inputs and FP16/FP32 output, using
 * CUTLASS FP8 GEMM with per-tensor dequantization scales:
 * <pre>
 *   C = dequant(A @ B) + bias
 *     = (scale_A * A_fp8) @ (scale_B * B_fp8) + bias
 * </pre>
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: A tensor (FP8 E4M3 quantized, stored as INT8) [M, K]</li>
 *   <li>1: B tensor (FP8 E4M3 quantized, stored as INT8) [K, N]</li>
 *   <li>2: scale_A (float scalar) - per-tensor dequantization scale for A</li>
 *   <li>3: scale_B (float scalar) - per-tensor dequantization scale for B</li>
 *   <li>4: bias (optional) [N]</li>
 * </ul>
 * <p>
 * Output: C tensor (FLOAT16) [M, N]
 * <p>
 * Integer arguments:
 * <ul>
 *   <li>0: fp8_format (0=E4M3, 1=E5M2, default: 0)</li>
 *   <li>1: transpose_a (0=no, 1=yes, default: 0)</li>
 *   <li>2: transpose_b (0=no, 1=yes, default: 0)</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class Fp8Matmul extends DynamicCustomOp {

    /** FP8 E4M3 format (4-bit exponent, 3-bit mantissa) - higher precision */
    public static final int FP8_E4M3 = 0;
    /** FP8 E5M2 format (5-bit exponent, 2-bit mantissa) - wider range */
    public static final int FP8_E5M2 = 1;

    @Getter private int fp8Format = FP8_E4M3;
    @Getter private boolean transposeA = false;
    @Getter private boolean transposeB = false;

    /**
     * SameDiff constructor with required inputs.
     *
     * @param sameDiff the SameDiff instance
     * @param a        FP8 quantized A matrix [M, K]
     * @param b        FP8 quantized B matrix [K, N]
     * @param scaleA   per-tensor scale for A
     * @param scaleB   per-tensor scale for B
     */
    public Fp8Matmul(SameDiff sameDiff, SDVariable a, SDVariable b,
                     SDVariable scaleA, SDVariable scaleB) {
        super(null, sameDiff, new SDVariable[]{a, b, scaleA, scaleB}, false);
        addIArgument((long) fp8Format, transposeA ? 1L : 0L, transposeB ? 1L : 0L);
    }

    /**
     * SameDiff constructor with bias.
     */
    public Fp8Matmul(SameDiff sameDiff, SDVariable a, SDVariable b,
                     SDVariable scaleA, SDVariable scaleB, SDVariable bias) {
        super(null, sameDiff, new SDVariable[]{a, b, scaleA, scaleB, bias}, false);
        addIArgument((long) fp8Format, transposeA ? 1L : 0L, transposeB ? 1L : 0L);
    }

    /**
     * Full SameDiff constructor with all options.
     */
    public Fp8Matmul(SameDiff sameDiff, SDVariable a, SDVariable b,
                     SDVariable scaleA, SDVariable scaleB, SDVariable bias,
                     int fp8Format, boolean transposeA, boolean transposeB) {
        super(null, sameDiff, bias != null ?
                new SDVariable[]{a, b, scaleA, scaleB, bias} :
                new SDVariable[]{a, b, scaleA, scaleB}, false);
        this.fp8Format = fp8Format;
        this.transposeA = transposeA;
        this.transposeB = transposeB;
        addIArgument((long) fp8Format, transposeA ? 1L : 0L, transposeB ? 1L : 0L);
    }

    /**
     * INDArray convenience constructor with defaults.
     */
    public Fp8Matmul(INDArray a, INDArray b, INDArray scaleA, INDArray scaleB) {
        super(null, new INDArray[]{a, b, scaleA, scaleB}, null);
        addIArgument((long) fp8Format, transposeA ? 1L : 0L, transposeB ? 1L : 0L);
    }

    /**
     * INDArray constructor.
     */
    public Fp8Matmul(INDArray a, INDArray b, INDArray scaleA, INDArray scaleB,
                     INDArray bias, INDArray output,
                     int fp8Format, boolean transposeA, boolean transposeB) {
        super(null, bias != null ?
                new INDArray[]{a, b, scaleA, scaleB, bias} :
                new INDArray[]{a, b, scaleA, scaleB},
                output != null ? new INDArray[]{output} : null);
        this.fp8Format = fp8Format;
        this.transposeA = transposeA;
        this.transposeB = transposeB;
        addIArgument((long) fp8Format, transposeA ? 1L : 0L, transposeB ? 1L : 0L);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.fp8Format = iArguments.get(0).intValue();
        if (iArguments.size() > 1) this.transposeA = iArguments.get(1) != 0;
        if (iArguments.size() > 2) this.transposeB = iArguments.get(2) != 0;
    }

    @Override
    public String opName() {
        return "fp8_matmul";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        // FP8 matmul always outputs FLOAT16 (or FLOAT for accumulation)
        return Collections.singletonList(DataType.HALF);
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
