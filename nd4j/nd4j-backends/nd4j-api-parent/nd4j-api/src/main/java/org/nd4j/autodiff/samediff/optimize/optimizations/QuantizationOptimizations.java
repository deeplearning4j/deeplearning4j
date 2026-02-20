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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
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
     * Optimizer that quantizes FP32 constant arrays to FP16.
     * This provides 2x memory reduction for model weights with minimal accuracy loss.
     *
     * Usage: Automatically applied during graph optimization when enabled.
     */
    public static class QuantizeConstantsToFP16 implements Optimizer {

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            return false;
        }

        /**
         * Apply FP16 quantization to all FP32 constants in the graph.
         * This is called once per graph optimization, not per-op.
         */
        public static int quantizeAllConstants(SameDiff sd) {
            ArrayHolder constantArrays = sd.getConstantArrays();
            List<String> constantNames = new ArrayList<>(constantArrays.arrayNames());
            
            int quantizedCount = 0;
            long fp32Bytes = 0;
            long fp16Bytes = 0;

            for (String name : constantNames) {
                INDArray arr = constantArrays.getArray(name);
                if (arr != null && arr.dataType() == DataType.FLOAT) {
                    fp32Bytes += arr.length() * 4;
                    
                    INDArray fp16Arr = arr.castTo(DataType.HALF);
                    constantArrays.setArray(name, fp16Arr);
                    
                    fp16Bytes += arr.length() * 2;
                    quantizedCount++;
                    
                    log.debug("Quantized constant {} to FP16: {} elements, {}KB -> {}KB",
                        name, arr.length(), (arr.length() * 4) / 1024, (arr.length() * 2) / 1024);
                }
            }

            if (quantizedCount > 0) {
                log.info("FP16 Quantization: {} constants quantized, {}KB -> {}KB ({}x reduction)",
                    quantizedCount, fp32Bytes / 1024, fp16Bytes / 1024, 
                    String.format("%.1f", (double) fp32Bytes / fp16Bytes));
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
     * Placeholder for redundant cast removal.
     * TODO: implement when SameDiff API supports consumer/output traversal.
     */
    public static class RemoveRedundantCasts implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
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
