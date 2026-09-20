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
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GgmlQMatMul;
import org.nd4j.linalg.api.ops.impl.transforms.custom.ModelOptNvfp4Linear;
import org.nd4j.linalg.api.ops.impl.transforms.custom.ModelOptFp8Linear;

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

    /** Distinct from GGUF .__q__: ModelOpt scales are independent tensors, never GGML blocks. */
    public static final String MODELOPT_BLOCK_SCALE = ".__modelopt_block_scale__";
    public static final String MODELOPT_GLOBAL_SCALE = ".__modelopt_global_scale__";
    public static final String MODELOPT_FP8_SCALE = ".__modelopt_fp8_scale__";
    public static final String MODELOPT_INPUT_SCALE = ".__modelopt_input_scale__";

    private QuantizedLinear() {}

    /**
     * Logical output dimension {@code N} for a GGUF linear weight. Runtime-quantized weights are
     * stored as packed bytes, so their INDArray shape is storage layout, not matrix shape.
     */
    public static int logicalOutputDim(Map<String, INDArray> weights, String ggufWeightName, INDArray weight) {
        return logicalDimension(weights, ggufWeightName, weight, 1, 0, "output");
    }

    /**
     * Logical input dimension {@code K} for a GGUF linear weight. Runtime-quantized weights are
     * stored as packed bytes, so their INDArray shape is storage layout, not matrix shape.
     */
    public static int logicalInputDim(Map<String, INDArray> weights, String ggufWeightName, INDArray weight) {
        return logicalDimension(weights, ggufWeightName, weight, 2, 1, "input");
    }

    private static int logicalDimension(Map<String, INDArray> weights, String ggufWeightName,
                                        INDArray weight, int metadataIndex, int denseShapeIndex,
                                        String dimensionName) {
        if (weight == null) {
            throw new IllegalArgumentException("Missing linear weight " + ggufWeightName);
        }
        INDArray meta = (ggufWeightName != null && weights != null)
                ? weights.get(ggufWeightName + ".__q__") : null;
        long value;
        if (meta != null) {
            if (meta.length() <= metadataIndex) {
                throw new IllegalArgumentException("Malformed quantized metadata for " + ggufWeightName
                        + ": expected [quantType,N,K], length=" + meta.length());
            }
            value = meta.getLong(metadataIndex);
        } else {
            long[] shape = weight.shape();
            if (shape == null || shape.length <= denseShapeIndex) {
                throw new IllegalArgumentException("Linear weight " + ggufWeightName
                        + " must have dense [N,K] shape, got " + java.util.Arrays.toString(shape));
            }
            value = shape[denseShapeIndex];
            if (denseShapeIndex == 1 && weights != null
                    && weights.containsKey(ggufWeightName + MODELOPT_BLOCK_SCALE)) {
                value = Math.multiplyExact(value, 2L);
            }
        }
        if (value <= 0 || value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Linear weight " + ggufWeightName + " has invalid logical "
                    + dimensionName + " dimension " + value);
        }
        return (int) value;
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
        if (weights != null && ggufWeightName != null) {
            boolean nvfp4 = weights.containsKey(ggufWeightName + MODELOPT_BLOCK_SCALE);
            boolean fp8 = weights.containsKey(ggufWeightName + MODELOPT_FP8_SCALE);
            if (nvfp4 || fp8) {
                if (nvfp4 && fp8 || weights.containsKey(ggufWeightName + ".__q__")) {
                    throw new IllegalArgumentException("Conflicting quantization metadata: " + ggufWeightName);
                }
                SDVariable scale = modelOptConstant(sd, weightVar.name(), weights, ggufWeightName,
                        nvfp4 ? MODELOPT_BLOCK_SCALE : MODELOPT_FP8_SCALE);
                SDVariable secondScale = modelOptConstant(sd, weightVar.name(), weights, ggufWeightName,
                        nvfp4 ? MODELOPT_GLOBAL_SCALE : MODELOPT_INPUT_SCALE);
                SDVariable result = nvfp4
                        ? new ModelOptNvfp4Linear(sd, activation, weightVar, scale, secondScale,
                                outputDtype == DataType.FLOAT).outputVariable()
                        : new ModelOptFp8Linear(sd, activation, weightVar, scale, secondScale,
                                outputDtype == DataType.FLOAT).outputVariable();
                return sd.updateVariableNameAndReference(result, name);
            }
        }
        INDArray meta = (ggufWeightName != null && weights != null) ? weights.get(ggufWeightName + ".__q__") : null;
        if (meta != null && weightVar != null && weightVar.dataType() == DataType.BYTE) {
            int quantType = (int) meta.getLong(0);
            long n = meta.getLong(1);
            long k = meta.getLong(2);
            // Runtime-quantized matmul accumulates in fp32. Keep that result in fp32 instead of
            // truncating to HALF; downstream norms/residuals and QLoRA zero-delta paths must not
            // see saturated +/-Inf activations from large packed-weight projections.
            int ggmlOutputDtype = GgmlQMatMul.OUTPUT_FLOAT32;
            // Use the generated namespace so validation and naming stay consistent with other
            // SameDiff importer operations. The packed weight keeps its own name for PEFT matching.
            return sd.nn().ggmlQMatMul(
                    name, activation, weightVar, quantType, n, k, ggmlOutputDtype);
        }
        return fp32Mmul(sd, name, activation, weightVar.permute(1, 0), computeDtype, outputDtype);
    }

    private static SDVariable modelOptConstant(SameDiff sd, String variableName,
            Map<String, INDArray> weights, String weightName, String suffix) {
        INDArray array = weights.get(weightName + suffix);
        if (array == null) {
            throw new IllegalArgumentException("Missing ModelOpt tensor " + weightName + suffix);
        }
        String name = variableName + suffix;
        SDVariable existing = sd.getVariable(name);
        return existing != null ? existing : sd.constant(name, array);
    }

    /**
     * Matmul using the importer-wide accumulation policy. Low-precision floating
     * operands accumulate in FP32 and the result is restored to the requested
     * model storage type.
     */
    public static SDVariable fp32Mmul(SameDiff sd, String name, SDVariable a, SDVariable b, DataType dtype) {
        return fp32Mmul(sd, name, a, b, dtype, dtype);
    }

    private static SDVariable fp32Mmul(SameDiff sd, String name, SDVariable a, SDVariable b,
                                       DataType computeDtype, DataType outputDtype) {
        DataType accumulationType = outputDtype == DataType.FLOAT
                ? outputDtype
                : GGMLDTypePolicy.accumulationType(computeDtype);
        boolean restoreOutputType = accumulationType != outputDtype;
        // Dense low-precision inference must use the same K recurrence for W=1
        // and W>1. Read original storage directly; do not materialize FP32 weights.
        boolean serialFma = computeDtype == DataType.HALF || computeDtype == DataType.BFLOAT16;
        String resultName = restoreOutputType ? name + "_accum" : name;
        SDVariable result;
        if (serialFma && accumulationType == DataType.FLOAT) {
            result = new Mmul(sd, a, b, MMulTranspose.allFalse(), Mmul.Arithmetic.SERIAL_FMA, DataType.FLOAT)
                    .outputVariable().rename(resultName);
        } else {
            SDVariable computeA = GGMLDTypePolicy.castTo(a, name + "_a_accum", accumulationType);
            SDVariable computeB = GGMLDTypePolicy.castTo(b, name + "_b_accum", accumulationType);
            result = serialFma
                    ? new Mmul(sd, computeA, computeB, MMulTranspose.allFalse(), Mmul.Arithmetic.SERIAL_FMA)
                            .outputVariable().rename(resultName)
                    : sd.mmul(resultName, computeA, computeB);
        }
        return restoreOutputType
                ? GGMLDTypePolicy.castTo(result, name, outputDtype)
                : result;
    }
}
