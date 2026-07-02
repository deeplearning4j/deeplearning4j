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
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.linalg.api.ops.impl.transforms.dtype.Cast;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.common.config.ND4JSystemProperties;
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
     * Optimizer that quantizes FP32 constant and variable arrays to FP16 or BF16.
     * This provides ~2x memory reduction for model weights with minimal accuracy loss.
     *
     * FP16 quantization is OFF by default (mixed-type execution not yet correct for all ops).
     * Enable with:
     * - {@code -Dnd4j.optimizer.fp16=true}
     *
     * BF16 is opt-in (requires hardware support):
     * - {@code -Dnd4j.optimizer.bf16=true}
     *
     * Only one mode can be active. BF16 takes precedence if both are set.
     *
     * Runs once on the first op encountered (quantizes all constants/variables at once).
     */
    public static class QuantizeConstantsToFP16 implements Optimizer {

        private boolean applied = false;

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (applied) return false;
            // FP16 defaults to OFF; enable with -Dnd4j.optimizer.fp16=true
            String fp16Prop = System.getProperty(ND4JSystemProperties.OPTIMIZER_FP16);
            boolean fp16Enabled = "true".equalsIgnoreCase(fp16Prop);
            boolean bf16Enabled = "true".equalsIgnoreCase(System.getProperty(ND4JSystemProperties.OPTIMIZER_BF16));
            if (!fp16Enabled && !bf16Enabled) {
                applied = true;
                return false;
            }
            // BF16 is opt-in and takes precedence over FP16 default
            applied = true;
            DataType targetType = bf16Enabled ? DataType.BFLOAT16 : DataType.HALF;
            String modeName = targetType == DataType.BFLOAT16 ? "BF16" : "FP16";

            int count = quantizeAllToType(sd, targetType);
            if (count > 0) {
                log.info("Full {} quantization: {} arrays converted to {}", modeName, count, targetType);
                return true;
            }
            return false;
        }

        /**
         * Convert all FP32 CONSTANT and VARIABLE arrays to HALF for full FP16 inference.
         */
        public static int quantizeAllToHalf(SameDiff sd) {
            return quantizeAllToType(sd, DataType.HALF);
        }

        /**
         * Convert all FP32 CONSTANT and VARIABLE arrays to BFLOAT16 for full BF16 inference.
         */
        public static int quantizeAllToBFloat16(SameDiff sd) {
            return quantizeAllToType(sd, DataType.BFLOAT16);
        }

        /**
         * Convert all FP32 CONSTANT and VARIABLE arrays to a target low-precision dtype.
         *
         * FP16 is broadly supported on CUDA tensor-core paths.
         * BF16 support depends on backend/hardware capabilities and is best used
         * when the runtime path has explicit BF16 kernels enabled.
         *
         * Integer-typed arrays (LONG, INT, etc.) are not affected.
         *
         * Minimum number of elements for an array to be quantized to low precision.
         * Only large 2D+ weight matrices benefit from FP16/BF16.
         * Small arrays (biases, normalization gammas/betas, scalars) stay FP32
         * because element-wise ops may not handle mixed HALF+FLOAT correctly
         * and the memory savings are negligible.
         */
        private static final long MIN_ELEMENTS_FOR_LOW_PRECISION = 1024;

        /**
         * Quantize all large FP32 constants/variables to a low-precision target type.
         *
         * <p>Constants are converted in-place to HALF/BFLOAT16 for ~2x memory savings.
         * Element-wise ops (add, mul, etc.) automatically upcast HALF→FLOAT via
         * {@code pickPairwiseResultType}, so no dequant Cast nodes are needed.
         * Matmul ops handle mixed HALF+FLOAT natively via cuBLAS.</p>
         */
        public static int quantizeAllToType(SameDiff sd, DataType targetType) {
            if (targetType != DataType.HALF && targetType != DataType.BFLOAT16) {
                throw new IllegalArgumentException("Unsupported low-precision target type: " + targetType);
            }
            ArrayHolder constantArrays = sd.getConstantArrays();
            ArrayHolder variableArrays = sd.getVariablesArrays();
            int quantizedCount = 0;
            int skippedCount = 0;
            long fp32Bytes = 0;
            long quantizedBytes = 0;
            String modeName = targetType == DataType.BFLOAT16 ? "BF16" : "FP16";

            for (String name : new ArrayList<>(constantArrays.arrayNames())) {
                INDArray arr = constantArrays.getArray(name);
                if (arr != null && arr.dataType() == DataType.FLOAT) {
                    if (arr.rank() >= 2 && arr.length() >= MIN_ELEMENTS_FOR_LOW_PRECISION) {
                        fp32Bytes += arr.length() * 4;
                        INDArray quantizedArr = arr.castTo(targetType);
                        constantArrays.removeArray(name);
                        constantArrays.setArray(name, quantizedArr);
                        quantizedBytes += arr.length() * targetType.width();
                        quantizedCount++;
                    } else {
                        skippedCount++;
                    }
                }
            }

            for (String name : new ArrayList<>(variableArrays.arrayNames())) {
                INDArray arr = variableArrays.getArray(name);
                if (arr != null && arr.dataType() == DataType.FLOAT) {
                    if (arr.rank() >= 2 && arr.length() >= MIN_ELEMENTS_FOR_LOW_PRECISION) {
                        fp32Bytes += arr.length() * 4;
                        INDArray quantizedArr = arr.castTo(targetType);
                        variableArrays.removeArray(name);
                        variableArrays.setArray(name, quantizedArr);
                        quantizedBytes += arr.length() * targetType.width();
                        quantizedCount++;
                    } else {
                        skippedCount++;
                    }
                }
            }

            if (quantizedCount > 0) {
                log.info("{} quantization: {} arrays quantized ({}MB -> {}MB, {}x reduction), {} small kept FP32",
                    modeName, quantizedCount, fp32Bytes / (1024 * 1024), quantizedBytes / (1024 * 1024),
                    String.format("%.1f", (double) fp32Bytes / Math.max(1, quantizedBytes)),
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
                // Don't remove identity casts whose output is a registered graph output —
                // the variable name must survive for DSP plan output mapping and output collection.
                List<String> graphOutputs = sd.outputs();
                if (graphOutputs != null && graphOutputs.contains(outputVar)) {
                    log.debug("Skipping identity cast removal for {} — output is a graph output", outputVar);
                    return false;
                }
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
     * Calibration-based activation INT8 quantization (MIGraphX: quantize_int8 two-phase).
     *
     * This optimizer inserts quantize→dequantize pairs around activation tensors that feed
     * matmul/conv2d inputs, using per-tensor symmetric INT8 quantization:
     *   scale = maxAbs(activation) / 127
     *   quantized = round(x / scale), clamped to [-127, 127], cast to INT8
     *   dequantized = cast(quantized, FP32) * scale
     *
     * This is a two-phase process:
     *   Phase 1 (calibration): run representative inputs through the graph, collect per-tensor
     *                          activation min/max at matmul inputs.
     *   Phase 2 (graph rewrite): insert quant→dequant pairs for each calibrated activation,
     *                             using the computed scale.
     *
     * This optimizer is OFF by default. Enable with: -Dnd4j.optimizer.int8.activations=true
     *
     * Usage:
     *   // Phase 1: calibrate
     *   Map&lt;String, QuantizationInfo&gt; scaleMap = QuantizeActivationsInt8.calibrate(sd, calibInputs);
     *   // Phase 2: rewrite graph
     *   SameDiff quantized = QuantizeActivationsInt8.rewriteWithScales(sd, scaleMap);
     *   // Or trigger via GraphOptimizer after setting system property + providing calibration data
     */
    public static class QuantizeActivationsInt8 implements Optimizer {

        /**
         * System property to enable activation quantization.
         * Default: off.
         */
        public static final String PROP_ENABLE = "nd4j.optimizer.int8.activations";

        // Calibration data: activation variable name → [min, max] observed across all calibration inputs
        private final Map<String, double[]> calibrationRanges;

        /**
         * Create an activation int8 quantizer with pre-collected calibration ranges.
         * @param calibrationRanges map of activation variable name → [minVal, maxVal]
         */
        public QuantizeActivationsInt8(Map<String, double[]> calibrationRanges) {
            this.calibrationRanges = calibrationRanges != null ? calibrationRanges : new HashMap<>();
        }

        /**
         * Default no-arg constructor — disabled (returns false for all ops).
         * Use the constructor with calibration data, or call calibrate() first.
         */
        public QuantizeActivationsInt8() {
            this.calibrationRanges = new HashMap<>();
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            // Only active when system property is set AND calibration data is available
            if (!Boolean.getBoolean(PROP_ENABLE)) return false;
            if (calibrationRanges.isEmpty()) return false;
            // Only fire for Mmul ops (matmul)
            if (!(op.getOp() instanceof Mmul)) return false;

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() < 1) return false;

            boolean applied = false;

            // Try to quantize the activation input (index 0, the dynamic input X)
            // Weight quantization is handled by the weight-only INT8 pass; this pass is for activations.
            String activationVarName = inputs.get(0);
            SDVariable activationVar = sd.getVariable(activationVarName);
            if (activationVar == null) return false;

            // Skip if already a CONSTANT or VARIABLE (those are weights, not activations)
            if (activationVar.getVariableType() == VariableType.CONSTANT
                    || activationVar.getVariableType() == VariableType.VARIABLE) {
                return false;
            }

            // Look up calibrated range for this activation
            double[] range = calibrationRanges.get(activationVarName);
            if (range == null) return false;  // No calibration data for this activation

            double minVal = range[0];
            double maxVal = range[1];
            double maxAbs = Math.max(Math.abs(minVal), Math.abs(maxVal));
            if (maxAbs < 1e-10) return false;  // Near-zero activation, skip

            float scale = (float) (maxAbs / 127.0);

            // Build: quantized = round(activation / scale), clamped to [-127, 127], cast INT8, cast FP32, mul scale
            // We emulate this via nd4j ops in SameDiff:
            //   x_scaled = div(activation, scale_const)
            //   x_rounded = round(x_scaled)    [no round op in SD — use floor(x+0.5)]
            //   x_clamped = clipByValue(x_rounded, -127, 127)
            //   x_int8 = cast(x_clamped, INT8)
            //   x_deq = cast(x_int8, FP32)
            //   x_out = mul(x_deq, scale_const)
            //
            // Then replace activation input of matmul with x_out.
            //
            // Note: This inserts the dequantized (still FP32 valued) tensor into the graph.
            // The quantization error is bounded by scale/2 ≈ maxAbs / (127*2).

            String suffix = "act_q_" + System.identityHashCode(op);
            String scaleConstName = "act_scale_" + suffix;

            // Create scale constant
            INDArray scaleArr = Nd4j.scalar(scale);
            sd.constant(scaleConstName, scaleArr);

            // Build the quant-dequant chain in SameDiff
            SDVariable actVar = sd.getVariable(activationVarName);
            SDVariable scaleConst = sd.getVariable(scaleConstName);

            // x / scale
            SDVariable xScaled = actVar.div(scaleConst).rename("xscaled_" + suffix);

            // round: approximate with floor(x + 0.5) since SD may not have round
            SDVariable halfConst = sd.constant("half_" + suffix, Nd4j.scalar(0.5f));
            SDVariable xPlusHalf = xScaled.add(halfConst);
            SDVariable xRounded = sd.math().floor("xrounded_" + suffix, xPlusHalf);

            // clamp to [-127, 127]
            SDVariable xClamped = sd.math().clipByValue("xclamped_" + suffix, xRounded, -127.0, 127.0);

            // cast to INT8 then back to FP32 (this is the lossy quantization step)
            SDVariable xInt8 = xClamped.castTo("xint8_" + suffix, DataType.INT8);
            SDVariable xDeq = xInt8.castTo("xdeq_" + suffix, DataType.FLOAT);

            // multiply back by scale
            SDVariable xOut = xDeq.mul(scaleConst).rename("xout_" + suffix);

            // Wire xOut as the new activation input to the matmul
            inputs.set(0, xOut.name());

            // Update variable tracking: xOut is now consumed by this matmul op
            Variable xOutVar = sd.getVariables().get(xOut.name());
            if (xOutVar != null) {
                if (xOutVar.getInputsForOp() == null) xOutVar.setInputsForOp(new ArrayList<>());
                if (!xOutVar.getInputsForOp().contains(op.getName())) {
                    xOutVar.getInputsForOp().add(op.getName());
                }
            }
            // Remove this matmul from original activation's consumer list
            Variable origActVariable = sd.getVariables().get(activationVarName);
            if (origActVariable != null && origActVariable.getInputsForOp() != null) {
                origActVariable.getInputsForOp().remove(op.getName());
            }

            log.debug("QuantizeActivationsInt8: inserted quant->dequant for activation '{}' (scale={})",
                    activationVarName, scale);
            applied = true;

            return applied;
        }

        /**
         * Phase 1: collect per-activation min/max ranges by running the graph with calibration inputs.
         *
         * For each matmul op in the graph, captures the range of the activation tensor (input[0])
         * across all calibration batches. The ranges are used in Phase 2 to compute quantization scales.
         *
         * @param sd            The SameDiff graph to calibrate
         * @param calibInputs   List of input maps (placeholder name → array), one per calibration batch
         * @param outputVarNames Output variable names required for execution
         * @return Map of activation variable name → [minVal, maxVal] observed across all calibration batches
         */
        public static Map<String, double[]> calibrate(SameDiff sd,
                                                      List<Map<String, INDArray>> calibInputs,
                                                      String... outputVarNames) {
            Map<String, double[]> ranges = new HashMap<>();

            // Collect the names of all matmul activation inputs (index 0 = the non-weight input)
            Set<String> activationVarNames = new LinkedHashSet<>();
            for (SameDiffOp sdOp : sd.getOps().values()) {
                if (!(sdOp.getOp() instanceof Mmul)) continue;
                List<String> ins = sdOp.getInputsToOp();
                if (ins == null || ins.size() < 1) continue;
                String actName = ins.get(0);
                SDVariable actVar = sd.getVariable(actName);
                // Only dynamic activations (ARRAY or PLACEHOLDER), not weights
                if (actVar != null
                        && actVar.getVariableType() != VariableType.CONSTANT
                        && actVar.getVariableType() != VariableType.VARIABLE) {
                    activationVarNames.add(actName);
                }
            }

            if (activationVarNames.isEmpty()) {
                log.warn("QuantizeActivationsInt8.calibrate: no matmul activation tensors found");
                return ranges;
            }

            // Build list of outputs to fetch: user's required outputs + all activation vars
            List<String> fetchVars = new ArrayList<>();
            if (outputVarNames != null) {
                fetchVars.addAll(Arrays.asList(outputVarNames));
            }
            // Add activation vars that exist as graph variables
            for (String name : activationVarNames) {
                if (!fetchVars.contains(name) && sd.hasVariable(name)) {
                    fetchVars.add(name);
                }
            }

            // Run each calibration batch and accumulate min/max per activation
            for (Map<String, INDArray> batch : calibInputs) {
                try {
                    Map<String, INDArray> results = sd.output(batch, fetchVars);
                    for (String actName : activationVarNames) {
                        INDArray arr = results.get(actName);
                        if (arr == null) continue;
                        double minVal = arr.minNumber().doubleValue();
                        double maxVal = arr.maxNumber().doubleValue();
                        double[] existing = ranges.get(actName);
                        if (existing == null) {
                            ranges.put(actName, new double[]{minVal, maxVal});
                        } else {
                            existing[0] = Math.min(existing[0], minVal);
                            existing[1] = Math.max(existing[1], maxVal);
                        }
                    }
                } catch (Exception e) {
                    log.warn("QuantizeActivationsInt8.calibrate: error executing batch: {}", e.getMessage());
                }
            }

            log.info("QuantizeActivationsInt8.calibrate: collected ranges for {} activations from {} batches",
                    ranges.size(), calibInputs.size());
            return ranges;
        }

        /**
         * Phase 2: rewrite the graph by inserting quant-dequant pairs for each calibrated activation.
         *
         * This is a convenience method that creates a QuantizeActivationsInt8 optimizer
         * from the given scale map and runs it through the GraphOptimizer with only this pass enabled.
         *
         * @param sd        The SameDiff graph to rewrite (will be dup'd internally)
         * @param scaleMap  Map of activation variable name → QuantizationInfo (from calibrate())
         * @return The rewritten SameDiff graph
         */
        public static SameDiff rewriteWithScales(SameDiff sd, Map<String, QuantizationInfo> scaleMap) {
            // Convert QuantizationInfo to [min, max] range representation (symmetric: min = -maxAbs)
            Map<String, double[]> rangeMap = new HashMap<>();
            for (Map.Entry<String, QuantizationInfo> e : scaleMap.entrySet()) {
                double maxAbs = e.getValue().scale * 127.0;
                rangeMap.put(e.getKey(), new double[]{-maxAbs, maxAbs});
            }

            QuantizeActivationsInt8 optimizer = new QuantizeActivationsInt8(rangeMap);
            List<org.nd4j.autodiff.samediff.optimize.OptimizerSet> optimizers = new ArrayList<>();
            optimizers.add(new org.nd4j.autodiff.samediff.optimize.OptimizerSet() {
                @Override
                public List<Optimizer> getOptimizers() {
                    return Collections.singletonList(optimizer);
                }
            });

            List<String> outputs = sd.outputs();
            if (outputs == null) outputs = Collections.emptyList();
            return org.nd4j.autodiff.samediff.optimize.GraphOptimizer.optimize(sd, outputs, optimizers);
        }
    }

}
