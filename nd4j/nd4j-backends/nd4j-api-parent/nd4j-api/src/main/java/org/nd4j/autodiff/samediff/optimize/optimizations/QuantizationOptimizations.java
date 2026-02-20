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

package org.nd4j.autodiff.samediff.optimize.optimizations;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.autodiff.samediff.serde.FlatBuffersMapper;
import org.nd4j.linalg.api.ops.impl.transforms.dtype.Cast;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Quantization-related optimizations for reducing model precision and improving performance.
 *
 * This class provides optimizations for:
 * - Quantizing weights from FP32 to FP16 (2x memory reduction)
 * - Fusing quantize/dequantize operations that cancel each other
 * - Removing redundant cast operations
 * - Mixed precision inference patterns
 */
@Slf4j
public class QuantizationOptimizations extends BaseOptimizerSet {

    /**
     * Optimizer that quantizes FP32 constant and variable arrays to FP16.
     * This provides 2x memory reduction for model weights with minimal accuracy loss.
     *
     * Gated by system property {@code -Dnd4j.optimizer.fp16=true}.
     * Runs once on the first op encountered (quantizes all constants/variables at once).
     */
    public static class QuantizeConstantsToFP16 implements Optimizer {

        private boolean applied = false;

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (applied) return false;
            if (!"true".equalsIgnoreCase(System.getProperty("nd4j.optimizer.fp16"))) {
                applied = true;
                return false;
            }
            applied = true;

            // Only apply FP16 to large models (decoder) — small models like
            // vision encoder and embed_tokens need FP32 precision.
            int minOps = Integer.parseInt(System.getProperty("nd4j.optimizer.fp16.minOps", "1000"));
            if (sd.getOps().size() < minOps) {
                log.info("FP16 skipped: model has {} ops (threshold {})", sd.getOps().size(), minOps);
                return false;
            }

            int count = quantizeAllToHalf(sd);
            if (count > 0) {
                log.info("Full FP16 quantization: {} arrays converted to HALF", count);
                return true;
            }
            return false;
        }

        /**
         * Convert all FP32 CONSTANT and VARIABLE arrays to HALF for full FP16 inference.
         *
         * All key ops support HALF natively:
         * - Matmul: cublasHgemm (tensor cores on compute 6.0+)
         * - FlashAttention: explicit float16 templates
         * - RMS norm: float16 template with FP32 internal reductions
         * - OnnxMultiHeadAttention: accepts HALF Q/K/V
         * - Swish/SwiGLU: element-wise ops support HALF
         *
         * Integer-typed arrays (LONG, INT, etc.) are not affected.
         */
        /**
         * Minimum number of elements for an array to be quantized to HALF.
         * Only large 2D+ weight matrices benefit from FP16 (tensor cores).
         * Small arrays (biases, normalization gammas/betas, scalars) stay FP32
         * because element-wise ops may not handle mixed HALF+FLOAT correctly
         * and the memory savings are negligible.
         */
        private static final long MIN_ELEMENTS_FOR_FP16 = 1024;

        public static int quantizeAllToHalf(SameDiff sd) {
            ArrayHolder constantArrays = sd.getConstantArrays();
            ArrayHolder variableArrays = sd.getVariablesArrays();
            int quantizedCount = 0;
            int skippedCount = 0;
            long fp32Bytes = 0;
            long fp16Bytes = 0;

            // Convert large FP32 constants (2D+ weight matrices)
            for (String name : new ArrayList<>(constantArrays.arrayNames())) {
                INDArray arr = constantArrays.getArray(name);
                if (arr != null && arr.dataType() == DataType.FLOAT) {
                    // Only quantize large 2D+ arrays (matmul weights)
                    // Skip 1D (biases, norms), scalars, and small arrays
                    if (arr.rank() >= 2 && arr.length() >= MIN_ELEMENTS_FOR_FP16) {
                        fp32Bytes += arr.length() * 4;
                        INDArray fp16Arr = arr.castTo(DataType.HALF);
                        constantArrays.setArray(name, fp16Arr);
                        fp16Bytes += arr.length() * 2;
                        quantizedCount++;
                    } else {
                        skippedCount++;
                    }
                }
            }

            // Convert large FP32 variables (e.g., embed_tokens weight)
            for (String name : new ArrayList<>(variableArrays.arrayNames())) {
                INDArray arr = variableArrays.getArray(name);
                if (arr != null && arr.dataType() == DataType.FLOAT) {
                    if (arr.rank() >= 2 && arr.length() >= MIN_ELEMENTS_FOR_FP16) {
                        fp32Bytes += arr.length() * 4;
                        INDArray fp16Arr = arr.castTo(DataType.HALF);
                        variableArrays.setArray(name, fp16Arr);
                        fp16Bytes += arr.length() * 2;
                        quantizedCount++;
                    } else {
                        skippedCount++;
                    }
                }
            }

            if (quantizedCount > 0) {
                log.info("FP16 weights: {} arrays quantized ({}MB -> {}MB, {}x), {} small arrays kept FP32",
                    quantizedCount, fp32Bytes / (1024 * 1024), fp16Bytes / (1024 * 1024),
                    String.format("%.1f", (double) fp32Bytes / Math.max(1, fp16Bytes)),
                    skippedCount);
            }

            return quantizedCount;
        }

    }

    /**
     * Quantizes FP32 constants to INT8 for 4x memory reduction.
     * Uses symmetric quantization (zero_point = 0) for better performance.
     *
     * Quantization formula: y = round(x / scale)
     * Dequantization formula: x = y * scale
     *
     * where scale = max_abs(x) / 127 (for INT8 range -127 to 127)
     *
     * Note: After quantization, use {@link #dequantizeAllConstants(SameDiff)} to restore FP32
     * for inference, or use {@link #quantizeToInt8(INDArray, QuantizationInfo)} with stored
     * QuantizationInfo for manual dequantization.
     */
    public static class QuantizeConstantsToINT8 implements Optimizer {

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            return false;
        }

        /**
         * Apply INT8 quantization to all FP32 constants in the graph.
         * Returns a map of constant name -> quantization info (scale, zero_point)
         *
         * @param sd SameDiff graph
         * @return map of constant names to quantization info
         */
        public static Map<String, QuantizationInfo> quantizeAllConstants(SameDiff sd) {
            ArrayHolder constantArrays = sd.getConstantArrays();
            List<String> constantNames = new ArrayList<>(constantArrays.arrayNames());

            Map<String, QuantizationInfo> quantizationInfo = new HashMap<>();
            int quantizedCount = 0;
            long fp32Bytes = 0;
            long int8Bytes = 0;

            for (String name : constantNames) {
                INDArray arr = constantArrays.getArray(name);
                if (arr != null && arr.dataType() == DataType.FLOAT) {
                    fp32Bytes += arr.length() * 4;

                    QuantizationInfo info = computeQuantizationInfo(arr);
                    INDArray int8Arr = quantizeToInt8(arr, info);
                    constantArrays.setArray(name, int8Arr);

                    int8Bytes += arr.length() * 1;
                    quantizedCount++;

                    quantizationInfo.put(name, info);

                    log.debug("Quantized constant {} to INT8: {} elements, {}KB -> {}KB",
                        name, arr.length(), (arr.length() * 4) / 1024, (arr.length() * 1) / 1024);
                }
            }

            if (quantizedCount > 0) {
                log.info("INT8 Quantization: {} constants quantized, {}KB -> {}KB ({}x reduction)",
                    quantizedCount, fp32Bytes / 1024, int8Bytes / 1024,
                    String.format("%.1f", (double) fp32Bytes / int8Bytes));
            }

            return quantizationInfo;
        }

        /**
         * Compute quantization info (scale and zero_point) for a given array.
         */
        public static QuantizationInfo computeQuantizationInfo(INDArray arr) {
            float maxAbs = arr.amaxNumber().floatValue();
            float scale = maxAbs > 0 ? maxAbs / 127.0f : 1.0f;
            return new QuantizationInfo(scale, 0);
        }

        /**
         * Quantize an array to INT8 using the given quantization info.
         */
        public static INDArray quantizeToInt8(INDArray arr, QuantizationInfo info) {
            INDArray scaled = arr.div(info.scale);
            // Round to nearest integer, clamp to INT8 range, then cast
            INDArray rounded = Nd4j.math().round(scaled);
            INDArray clamped = Nd4j.math().clipByValue(rounded, -127, 127);
            return clamped.castTo(DataType.INT8);
        }

        /**
         * Dequantize an INT8 array back to FP32 using the given quantization info.
         */
        public static INDArray dequantizeFromInt8(INDArray arr, QuantizationInfo info) {
            INDArray floatArr = arr.castTo(DataType.FLOAT);
            return floatArr.mul(info.scale);
        }

        /**
         * Apply INT8 quantization and store scales as separate constants.
         * This allows later retrieval for dequantization.
         *
         * @param sd SameDiff graph
         * @return map of constant names to their quantization info
         */
        public static Map<String, QuantizationInfo> quantizeAllConstantsWithScales(SameDiff sd) {
            ArrayHolder constantArrays = sd.getConstantArrays();
            List<String> constantNames = new ArrayList<>(constantArrays.arrayNames());

            Map<String, QuantizationInfo> quantizationInfo = new HashMap<>();
            int quantizedCount = 0;
            long fp32Bytes = 0;
            long int8Bytes = 0;

            for (String name : constantNames) {
                INDArray arr = constantArrays.getArray(name);
                if (arr != null && arr.dataType() == DataType.FLOAT) {
                    fp32Bytes += arr.length() * 4;

                    QuantizationInfo info = computeQuantizationInfo(arr);
                    INDArray int8Arr = quantizeToInt8(arr, info);
                    constantArrays.setArray(name, int8Arr);

                    // Store scale as a separate constant for later retrieval
                    String scaleName = name + "_quant_scale";
                    sd.constant(scaleName, Nd4j.scalar(info.scale));

                    int8Bytes += arr.length();
                    quantizedCount++;
                    quantizationInfo.put(name, info);

                    log.debug("Quantized {} to INT8: scale={}", name, info.scale);
                }
            }

            if (quantizedCount > 0) {
                log.info("INT8 Quantization: {} constants, {}KB -> {}KB ({}x reduction)",
                    quantizedCount, fp32Bytes / 1024, int8Bytes / 1024,
                    String.format("%.1f", (double) fp32Bytes / int8Bytes));
            }

            return quantizationInfo;
        }

        /**
         * Get quantization info for a constant by name.
         * @param sd SameDiff graph
         * @param name constant name
         * @return quantization info, or null if not found
         */
        public static QuantizationInfo getQuantizationInfo(SameDiff sd, String name) {
            String scaleName = name + "_quant_scale";
            INDArray scaleArr = sd.getConstantArrays().getArray(scaleName);
            if (scaleArr != null) {
                return new QuantizationInfo(scaleArr.getFloat(0), 0);
            }
            return null;
        }
    }

    /**
     * Holds quantization parameters for INT8 quantization.
     */
    public static class QuantizationInfo {
        public final float scale;
        public final int zeroPoint;

        public QuantizationInfo(float scale, int zeroPoint) {
            this.scale = scale;
            this.zeroPoint = zeroPoint;
        }

        @Override
        public String toString() {
            return "QuantizationInfo{scale=" + scale + ", zeroPoint=" + zeroPoint + "}";
        }
    }

    /**
     * Placeholder for DequantizeLinear/QuantizeLinear pair fusion.
     * TODO: implement when SameDiff API supports consumer/output traversal.
     */
    public static class FuseDequantizeQuantizePair implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            return false;
        }
    }

    /**
     * Removes redundant cast operations:
     * 1. Identity casts where input dtype == output dtype (FLOAT→FLOAT)
     * 2. Chained casts where cast(cast(x, A), B) can be replaced with cast(x, B)
     */
    public static class RemoveRedundantCasts implements Optimizer {

        private static final Set<Class<? extends DifferentialFunction>> APPLICABLE_OPS = new HashSet<>();
        static {
            APPLICABLE_OPS.add(Cast.class);
        }

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return APPLICABLE_OPS;
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Cast)) {
                return false;
            }

            Cast castOp = (Cast) op.getOp();
            List<String> inputs = op.getInputsToOp();
            List<String> outputs = op.getOutputsOfOp();
            if (inputs == null || inputs.isEmpty() || outputs == null || outputs.isEmpty()) {
                return false;
            }

            String inputVar = inputs.get(0);
            String outputVar = outputs.get(0);

            // Determine target dtype from Cast op's iArguments (FlatBuffers byte)
            long[] iArgs = castOp.iArgs();
            if (iArgs == null || iArgs.length == 0) {
                return false;
            }
            DataType outputDtype = FlatBuffersMapper.getDataTypeFromByte((byte) iArgs[0]);

            // Determine input dtype — try multiple sources
            DataType inputDtype = null;
            SDVariable inputSdVar = sd.getVariable(inputVar);
            if (inputSdVar != null) {
                inputDtype = inputSdVar.dataType();
            }
            // For ARRAY types, dataType() often returns null — try to infer from producer op
            if (inputDtype == null) {
                Variable inputVariable = sd.getVariables().get(inputVar);
                if (inputVariable != null) {
                    String producerOpName = inputVariable.getOutputOfOp();
                    if (producerOpName != null) {
                        SameDiffOp producerOp = sd.getOps().get(producerOpName);
                        if (producerOp != null && producerOp.getOp() instanceof Cast) {
                            // Producer is also a cast — its output type is its iArgs[0]
                            long[] producerIArgs = ((Cast) producerOp.getOp()).iArgs();
                            if (producerIArgs != null && producerIArgs.length > 0) {
                                inputDtype = FlatBuffersMapper.getDataTypeFromByte((byte) producerIArgs[0]);
                            }
                        }
                    }
                }
                // If input is a CONSTANT or VARIABLE, get dtype from the array
                if (inputDtype == null && inputSdVar != null) {
                    VariableType vt = inputSdVar.getVariableType();
                    if (vt == VariableType.CONSTANT) {
                        INDArray arr = constantArrays.getArray(inputVar);
                        if (arr != null) inputDtype = arr.dataType();
                    } else if (vt == VariableType.VARIABLE) {
                        INDArray arr = variablesArrays.getArray(inputVar);
                        if (arr != null) inputDtype = arr.dataType();
                    }
                }
            }

            // Case 1: Identity cast (input dtype == output dtype)
            if (inputDtype != null && inputDtype == outputDtype) {
                OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, inputVar);
                OptimizationUtils.removeOp(sd, helper, op.getName());
                OptimizationUtils.removeVariable(sd, helper, outputVar);
                log.debug("Removed identity cast {} ({} → {})", op.getName(), inputDtype, outputDtype);
                return true;
            }

            // Case 2: Chained casts — cast(cast(x, A), B) → cast(x, B)
            Variable inputVariable = helper != null ? helper.getVariable(inputVar) : sd.getVariables().get(inputVar);
            if (inputVariable != null) {
                String producerOpName = inputVariable.getOutputOfOp();
                if (producerOpName != null) {
                    SameDiffOp producerOp = sd.getOps().get(producerOpName);
                    if (producerOp != null && producerOp.getOp() instanceof Cast) {
                        // Check if the intermediate cast output is only used by this cast
                        List<String> intermediateUsers = inputVariable.getInputsForOp();
                        if (intermediateUsers != null && intermediateUsers.size() == 1) {
                            // Rewire: this cast now takes the input of the inner cast
                            String innerInput = producerOp.getInputsToOp().get(0);
                            List<String> newInputs = new ArrayList<>(inputs);
                            newInputs.set(0, innerInput);
                            op.setInputsToOp(newInputs);

                            // Update variable tracking
                            Variable innerInputVar = helper != null ? helper.getVariable(innerInput) : sd.getVariables().get(innerInput);
                            if (innerInputVar != null) {
                                List<String> usedBy = innerInputVar.getInputsForOp();
                                if (usedBy != null && !usedBy.contains(op.getName())) {
                                    usedBy.add(op.getName());
                                }
                            }

                            // Remove inner cast
                            OptimizationUtils.removeOp(sd, helper, producerOpName);
                            OptimizationUtils.removeVariable(sd, helper, inputVar);
                            log.debug("Fused chained casts: {} absorbed into {}", producerOpName, op.getName());
                            return true;
                        }
                    }
                }
            }

            return false;
        }
    }

    /**
     * Placeholder for FP16 inference optimization.
     * Use {@link QuantizeConstantsToFP16#quantizeAllConstants(SameDiff)} instead.
     */
    public static class OptimizeConstantsForInference implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            return false;
        }
    }

    /**
     * Placeholder for backward compatibility.
     * @deprecated Use {@link QuantizeConstantsToFP16} or {@link OptimizeConstantsForInference} instead.
     */
    @Deprecated
    public static class QuantizePlaceholder implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            return false;
        }
    }
}
