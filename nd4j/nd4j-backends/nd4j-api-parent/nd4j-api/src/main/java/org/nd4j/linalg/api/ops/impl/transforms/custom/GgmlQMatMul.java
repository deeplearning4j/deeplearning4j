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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * GgmlQMatMul — Runtime quantized matmul against GGML-packed weights.
 *
 * Performs:  out[m,n] = sum_k  A[m,k] * dequant(W[n,k])
 * with fp32 accumulation, entirely within the kernel — no import-time dequantization.
 *
 * This is the single biggest decode-bandwidth improvement for GGUF quantized models:
 * instead of dequantizing all weights at load time (paying the BW cost repeatedly),
 * we read the packed bytes once per forward pass and dequantize on-the-fly.
 *
 * <b>Supported quantization formats (v1):</b>
 * <ul>
 *   <li>GGML_QUANT_Q8_0  (ggmlType=4)  — 34 bytes/32 elements</li>
 *   <li>GGML_QUANT_Q4_K  (ggmlType=8)  — 144 bytes/256 elements (covers Q4_K_M models)</li>
 *   <li>GGML_QUANT_Q6_K  (ggmlType=10) — 210 bytes/256 elements</li>
 * </ul>
 *
 * <b>Weight orientation:</b><br>
 * packedWeights encodes a logical weight matrix W of shape [N, K] in row-major order:
 * each of the N rows contains (K / blockSize) quantized blocks along the K dimension.
 * This matches GGUF storage after reverseShape() in GGMLToSameDiffConverter, and the
 * math out[m,n] = sum_k A[m,k] * W[n,k] is equivalent to A @ W^T (as used by
 * LLaMAArchitecture via weight.permute(1,0) before sd.mmul()).
 *
 * <b>Inputs:</b>
 * <ol>
 *   <li>activations — FLOAT32 or FLOAT16, rank 2 [M,K] or rank 3 [B,S,K]</li>
 *   <li>packedWeights — INT8 (raw GGML byte buffer), rank 1, length = N*(K/blockSize)*bytesPerBlock</li>
 * </ol>
 *
 * <b>Integer args:</b>
 * <ol>
 *   <li>ggmlType    — GgmlQuantType enum value (4, 8, or 10)</li>
 *   <li>N           — number of output columns (weight rows)</li>
 *   <li>K           — inner dimension</li>
 *   <li>outputDtype — 0 = FLOAT32, 1 = FLOAT16</li>
 * </ol>
 *
 * <b>Output:</b>  [M,N] or [B,S,N]
 */
public class GgmlQMatMul extends DynamicCustomOp {

    /** GgmlQuantType enum values matching GgmlQuantType in ggml_dequantize.h */
    public static final int GGML_QUANT_Q8_0 = 4;
    public static final int GGML_QUANT_Q4_K = 8;
    public static final int GGML_QUANT_Q6_K = 10;

    /** outputDtype constants */
    public static final int OUTPUT_FLOAT32 = 0;
    public static final int OUTPUT_FLOAT16 = 1;

    public GgmlQMatMul() {
        // no-arg constructor for serialization
    }

    /**
     * SameDiff constructor.
     *
     * @param sd           SameDiff instance
     * @param activations  SDVariable [M,K] or [B,S,K], FLOAT32 or FLOAT16
     * @param packedWeights SDVariable INT8 rank-1 buffer of raw GGML blocks
     * @param ggmlType     quantization type (GGML_QUANT_Q8_0, GGML_QUANT_Q4_K, or GGML_QUANT_Q6_K)
     * @param N            number of weight rows (output columns)
     * @param K            inner dimension
     * @param outputDtype  OUTPUT_FLOAT32 or OUTPUT_FLOAT16
     */
    public GgmlQMatMul(SameDiff sd, SDVariable activations, SDVariable packedWeights,
                       int ggmlType, long N, long K, int outputDtype) {
        super(null, sd, new SDVariable[]{activations, packedWeights});
        addIArgument(ggmlType, N, K, outputDtype);
    }

    /**
     * Eager (INDArray) constructor for the generated fluent ND API.
     */
    public GgmlQMatMul(INDArray activations, INDArray packedWeights,
                       int ggmlType, long N, long K, int outputDtype) {
        super(null, new INDArray[]{activations, packedWeights}, null);
        addIArgument(ggmlType, N, K, outputDtype);
    }

    /**
     * Eager (INDArray) constructor.
     *
     * @param activations   FLOAT32 or FLOAT16, rank 2 [M,K] or rank 3 [B,S,K]
     * @param packedWeights INT8 rank-1 raw GGML block bytes
     * @param ggmlType      quantization type constant
     * @param N             number of weight rows
     * @param K             inner dimension
     * @param outputDtype   OUTPUT_FLOAT32 or OUTPUT_FLOAT16
     * @return              output INDArray [M,N] or [B,S,N]
     */
    public static INDArray exec(INDArray activations, INDArray packedWeights,
                                int ggmlType, long N, long K, int outputDtype) {
        Preconditions.checkArgument(
            ggmlType == GGML_QUANT_Q8_0 || ggmlType == GGML_QUANT_Q4_K || ggmlType == GGML_QUANT_Q6_K,
            "GgmlQMatMul: unsupported ggmlType %s (supported: 4=Q8_0, 8=Q4_K, 10=Q6_K)", ggmlType);
        Preconditions.checkArgument(outputDtype == OUTPUT_FLOAT32 || outputDtype == OUTPUT_FLOAT16,
            "GgmlQMatMul: outputDtype must be 0 (FP32) or 1 (FP16), got %s", outputDtype);

        GgmlQMatMul op = new GgmlQMatMul();
        op.addInputArgument(activations, packedWeights);
        op.addIArgument(ggmlType, N, K, outputDtype);
        return Nd4j.exec(op)[0];
    }

    /**
     * Returns the bytes-per-block for a given ggmlType.
     * Returns -1 for unsupported types.
     */
    public static int bytesPerBlock(int ggmlType) {
        switch (ggmlType) {
            case GGML_QUANT_Q8_0: return 34;
            case GGML_QUANT_Q4_K: return 144;
            case GGML_QUANT_Q6_K: return 210;
            default: return -1;
        }
    }

    /**
     * Returns the elements-per-block for a given ggmlType.
     * Returns -1 for unsupported types.
     */
    public static int elemsPerBlock(int ggmlType) {
        switch (ggmlType) {
            case GGML_QUANT_Q8_0: return 32;
            case GGML_QUANT_Q4_K: return 256;
            case GGML_QUANT_Q6_K: return 256;
            default: return -1;
        }
    }

    /**
     * Computes the expected packed byte buffer length for a weight matrix [N, K].
     */
    public static long expectedPackedBytes(int ggmlType, long N, long K) {
        int bpb = bytesPerBlock(ggmlType);
        int epb = elemsPerBlock(ggmlType);
        if (bpb < 0 || epb < 0) return -1L;
        if (K % epb != 0) return -1L;
        return N * (K / epb) * bpb;
    }

    @Override
    public String opName() {
        return "ggml_qmatmul";
    }

    /**
     * Backprop through ggml_qmatmul.
     *
     * <p>Returns gradients for [activations, packedWeights].  The packed weight is a
     * frozen CONSTANT, so its gradient slot is filled with {@code zerosLike(packedWeights)}
     * and is never consumed by the optimiser.  The gradient for the activations is
     * computed via {@link GgmlQMatMulBp}.
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        SDVariable gradOut       = gradients.get(0);
        SDVariable activations   = arg(0);
        SDVariable packedWeights = arg(1);

        long[] iArgsArr = iArgs();
        int quantType = (iArgsArr != null && iArgsArr.length > 0) ? (int) iArgsArr[0] : 0;
        long N        = (iArgsArr != null && iArgsArr.length > 1) ? iArgsArr[1] : 0L;
        long K        = (iArgsArr != null && iArgsArr.length > 2) ? iArgsArr[2] : 0L;

        GgmlQMatMulBp bpOp = new GgmlQMatMulBp(sameDiff, activations, packedWeights, gradOut,
            quantType, N, K);
        SDVariable gradActivations = bpOp.outputVariables()[0];

        // Frozen weight — return zeros placeholder
        SDVariable gradPacked = sameDiff.zerosLike(packedWeights);

        return Arrays.asList(gradActivations, gradPacked);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        // outputDtype: INT_ARG(3)
        // If iargs available, use them; otherwise default to FLOAT
        if (iArguments != null && iArguments.size() >= 4) {
            int outputDtype = iArguments.get(3).intValue();
            return Collections.singletonList(outputDtype == OUTPUT_FLOAT16 ? DataType.HALF : DataType.FLOAT);
        }
        return Collections.singletonList(DataType.FLOAT);
    }
}
