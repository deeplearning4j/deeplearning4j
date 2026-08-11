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

package org.nd4j.ggml.architecture;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GgmlQMatMul;

import java.util.Map;

/**
 * Shared quantized-aware linear projection for GGUF architectures.
 *
 * <p>When a weight is imported with {@code ConversionOptions.RUNTIME_QUANTIZED_MATMUL} it is stored
 * as a packed INT8 ({@link DataType#BYTE}) tensor with a companion {@code "<ggufName>.__q__"}
 * {@code LONG[3] = [quantType, N, K]} entry in the raw weights map (see
 * {@code GGMLToSameDiffConverter}). This helper emits a single {@code ggml_qmatmul} op that
 * dequantizes the packed weight on the fly (fp32 accumulation). Dense floating weights use a
 * plain fp32-accumulated matmul with the weight permuted {@code [N,K] -> [K,N]}.</p>
 *
 * <p>This is what makes runtime-quantized inference and QLoRA (LoRA attached to {@code ggml_qmatmul})
 * work across all architectures. The dispatch previously existed only as an unused private method
 * in {@code LLaMAArchitecture}, so no architecture actually emitted {@code ggml_qmatmul}.</p>
 *
 * <p><b>Naming:</b> the {@code .__q__} companion is keyed by the GGUF tensor name (the key the
 * architecture used to read the weight from the map), while the weight is registered in the graph
 * under an architecture-specific variable name. Callers therefore pass BOTH the registered weight
 * variable and the GGUF name used for the map lookup.</p>
 */
public final class QuantizedLinear {

    private QuantizedLinear() {}

    /**
     * Logical output dimension {@code N} for a GGUF linear weight. Runtime-quantized weights are
     * stored as packed bytes, so their INDArray shape is storage layout, not matrix shape.
     */
    public static int logicalOutputDim(Map<String, INDArray> weights, String ggufWeightName, INDArray weight) {
        INDArray meta = (ggufWeightName != null && weights != null) ? weights.get(ggufWeightName + ".__q__") : null;
        if (meta != null) {
            long n = meta.getLong(1);
            if (n > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Quantized weight output dimension exceeds int range: " + n);
            }
            return (int) n;
        }
        return (int) weight.shape()[0];
    }

    /**
     * Quantized-aware linear projection: {@code activation @ weight^T}.
     *
     * @param sd             the SameDiff graph
     * @param name           output variable name
     * @param activation     input activations {@code [.., K]}
     * @param weightVar      the registered weight variable — BYTE packed when quantized, else FLOAT {@code [N,K]}
     * @param weights        the raw weights map (holds the {@code "<ggufWeightName>.__q__"} companion)
     * @param ggufWeightName the GGUF tensor name used to look up the weight (key for the {@code .__q__} companion); may be null
     * @param dtype          the model working dtype
     * @return               output {@code [.., N]}
     */
    public static SDVariable matMul(SameDiff sd, String name, SDVariable activation, SDVariable weightVar,
                                    Map<String, INDArray> weights, String ggufWeightName, DataType dtype) {
        return matMul(sd, name, activation, weightVar, weights, ggufWeightName, dtype, dtype);
    }

    /**
     * Quantized-aware linear projection that keeps the result in FLOAT.
     *
     * <p>Use this for logits and other numerically sensitive terminal projections where storing the
     * fp32-accumulated result back to HALF can overflow before the consumer gets a chance to upcast.</p>
     */
    public static SDVariable matMulFloatOutput(SameDiff sd, String name, SDVariable activation, SDVariable weightVar,
                                               Map<String, INDArray> weights, String ggufWeightName, DataType computeDtype) {
        return matMul(sd, name, activation, weightVar, weights, ggufWeightName, computeDtype, DataType.FLOAT);
    }

    private static SDVariable matMul(SameDiff sd, String name, SDVariable activation, SDVariable weightVar,
                                     Map<String, INDArray> weights, String ggufWeightName,
                                     DataType computeDtype, DataType outputDtype) {
        INDArray meta = (ggufWeightName != null && weights != null) ? weights.get(ggufWeightName + ".__q__") : null;
        if (meta != null && weightVar != null && weightVar.dataType() == DataType.BYTE) {
            int quantType = (int) meta.getLong(0);
            long n = meta.getLong(1);
            long k = meta.getLong(2);
            // Runtime-quantized matmul accumulates in fp32. Keep that result in fp32 instead of
            // truncating to HALF; downstream norms/residuals and QLoRA zero-delta paths must not
            // see saturated +/-Inf activations from large packed-weight projections.
            int ggmlOutputDtype = GgmlQMatMul.OUTPUT_FLOAT32;
            SDVariable result = new GgmlQMatMul(sd, activation, weightVar, quantType, n, k, ggmlOutputDtype).outputVariable();
            // Rename to the requested output name; the packed weightVar keeps its own name so PEFT
            // can match it against target-module patterns and attach a LoRA residual.
            return sd.identity(name, result);
        }
        return fp32Mmul(sd, name, activation, weightVar.permute(1, 0), computeDtype, outputDtype);
    }

    /**
     * Whether this floating storage dtype should use FP32 accumulation for numerically
     * sensitive operations.
     */
    public static boolean requiresFp32Accumulation(DataType dataType) {
        return dataType == DataType.HALF
                || dataType == DataType.BFLOAT16
                || dataType == DataType.FLOAT8
                || dataType == DataType.FLOAT8_E5M2;
    }

    /**
     * FP32-accumulated matmul. Upcasts low-precision floating operands to FLOAT to avoid
     * overflow on long dot products, then casts the result back. For FLOAT inputs it is
     * a plain matmul.
     */
    public static SDVariable fp32Mmul(SameDiff sd, String name, SDVariable a, SDVariable b, DataType dtype) {
        return fp32Mmul(sd, name, a, b, dtype, dtype);
    }

    private static SDVariable fp32Mmul(SameDiff sd, String name, SDVariable a, SDVariable b,
                                       DataType computeDtype, DataType outputDtype) {
        boolean outputFloat = outputDtype == DataType.FLOAT;
        boolean needsFloatCompute = outputFloat || requiresFp32Accumulation(computeDtype);
        if (needsFloatCompute) {
            SDVariable aF32 = a.dataType() == DataType.FLOAT ? a : a.castTo(name + "_a_f32", DataType.FLOAT);
            SDVariable bF32 = b.dataType() == DataType.FLOAT ? b : b.castTo(name + "_b_f32", DataType.FLOAT);
            SDVariable result = sd.mmul(outputFloat ? name : name + "_f32", aF32, bF32);
            return outputFloat ? result : result.castTo(name, outputDtype);
        }
        return sd.mmul(name, a, b);
    }
}
