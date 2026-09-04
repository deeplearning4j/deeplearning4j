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

package org.nd4j.ggml.quantization;

import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GGMLDequantize;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Factory for creating dequantizers based on quantization type.
 */
public class DequantizerFactory {

    private static final Logger log = LoggerFactory.getLogger(DequantizerFactory.class);
    private static final Map<GGMLDataType, Dequantizer> dequantizers = new ConcurrentHashMap<>();

    static {
        // Register built-in dequantizers
        register(new Q4_0Dequantizer());
        register(new Q4_1Dequantizer());
        register(new Q5_0Dequantizer());
        register(new Q5_1Dequantizer());
        register(new Q8_0Dequantizer());
        register(new Q2_KDequantizer());
        register(new Q3_KDequantizer());
        register(new Q4_KDequantizer());
        register(new Q5_KDequantizer());
        register(new Q6_KDequantizer());
        register(new Q8_KDequantizer());
        // Importance quantization types
        register(new IQ1_SDequantizer());
        register(new IQ1_MDequantizer());
        register(new IQ2_XXSDequantizer());
        register(new IQ2_XSDequantizer());
        register(new IQ2_SDequantizer());
        register(new IQ3_XXSDequantizer());
        register(new IQ3_SDequantizer());
        register(new IQ4_NLDequantizer());
        register(new IQ4_XSDequantizer());
        // Ternary quantization types
        register(new TQ1_0Dequantizer());
        register(new TQ2_0Dequantizer());
    }

    private DequantizerFactory() {
        // Utility class
    }

    /**
     * Register a dequantizer
     */
    public static void register(Dequantizer dequantizer) {
        dequantizers.put(dequantizer.getQuantType(), dequantizer);
    }

    /**
     * Get a dequantizer for the given type
     *
     * @throws IllegalArgumentException if no dequantizer is available
     */
    public static Dequantizer getDequantizer(GGMLDataType type) {
        Dequantizer dequantizer = dequantizers.get(type);
        if (dequantizer == null) {
            throw new IllegalArgumentException("No dequantizer available for type: " + type);
        }
        return dequantizer;
    }

    /**
     * Check if a dequantizer is available for the given type
     */
    public static boolean hasDequantizer(GGMLDataType type) {
        return dequantizers.containsKey(type);
    }

    /**
     * Get all registered quantization types
     */
    public static Set<GGMLDataType> getSupportedTypes() {
        return dequantizers.keySet();
    }

    /**
     * Dequantize data to float array
     */
    public static float[] dequantize(byte[] data, GGMLDataType type, long numElements) {
        return getDequantizer(type).dequantize(data, numElements);
    }

    /**
     * Dequantize data to INDArray using the native ggml_dequantize op.
     * Dequantizes directly to the target type when the native op supports it
     * (F32, F16, BF16). Other floating-point storage types are produced by an
     * explicit cast from the native F32 result; the native op is never given an
     * unsupported output type. Quantized formats without a native implementation
     * use their registered Java dequantizer as an explicit route. Native execution
     * failures are surfaced and are never silently retried through another path.
     */
    public static INDArray dequantizeToArray(byte[] data, GGMLDataType type, long[] shape, DataType targetType) {
        if (targetType == null || !targetType.isFPType()) {
            throw new IllegalArgumentException("GGML dequantization requires a floating-point target type, got: "
                    + targetType);
        }

        DataType nativeTargetType = directNativeTargetType(targetType);
        int nativeType = nativeQuantType(type);
        if (nativeType >= 0) {
            INDArray nativeResult = Nd4j.createUninitialized(nativeTargetType, shape);
            boolean dequantized = false;
            try (INDArray rawBytes = Nd4j.create(data, new long[]{data.length}, DataType.INT8)) {
                dequantizeInto(rawBytes, type, shape, nativeResult);
                dequantized = true;
            } finally {
                if (!dequantized) {
                    nativeResult.close();
                }
            }

            if (nativeTargetType == targetType) {
                return nativeResult;
            }
            try {
                return nativeResult.castTo(targetType);
            } finally {
                nativeResult.close();
            }
        }

        // Explicit Java implementation route for quantized formats without a native kernel.
        INDArray fp32 = getDequantizer(type).dequantizeToArray(data, shape, DataType.FLOAT);
        if (targetType == DataType.FLOAT) {
            return fp32;
        }
        try {
            return fp32.castTo(targetType);
        } finally {
            fp32.close();
        }
    }

    /**
     * Returns whether {@link #dequantizeInto(INDArray, GGMLDataType, long[], INDArray)}
     * can execute the type through the native op.
     */
    public static boolean supportsNativeDequantization(GGMLDataType type) {
        return nativeQuantType(type) >= 0;
    }

    /**
     * Dequantize into caller-owned buffers without allocating an input cast, output array,
     * or output shape. This is the reusable-buffer path for streamed model ingestion.
     */
    public static void dequantizeInto(INDArray rawBytes, GGMLDataType type,
                                      long[] shape, INDArray output) {
        if (rawBytes == null || rawBytes.dataType() != DataType.INT8) {
            throw new IllegalArgumentException("Native GGML dequantization requires an INT8 input array");
        }
        if (rawBytes.isView()) {
            throw new IllegalArgumentException("Native GGML dequantization input must own a contiguous buffer");
        }
        if (output == null || !output.dataType().isFPType()
                || directNativeTargetType(output.dataType()) != output.dataType()) {
            throw new IllegalArgumentException(
                    "Native GGML dequantization output must be FLOAT, HALF, or BFLOAT16");
        }
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Native GGML dequantization requires a non-empty shape");
        }
        long expectedLength = 1;
        for (long dimension : shape) {
            if (dimension < 0) {
                throw new IllegalArgumentException("Negative GGML dequantization dimension: " + dimension);
            }
            expectedLength = Math.multiplyExact(expectedLength, dimension);
        }
        if (output.length() != expectedLength) {
            throw new IllegalArgumentException("GGML dequantization output length mismatch: expected "
                    + expectedLength + " but got " + output.length());
        }
        if (output.isView() || !Arrays.equals(shape, output.shape())) {
            throw new IllegalArgumentException("Native GGML dequantization output must own a contiguous buffer "
                    + "with shape " + Arrays.toString(shape));
        }
        long requiredInputBytes = type.calculateStorageBytes(expectedLength);
        if (rawBytes.length() < requiredInputBytes) {
            throw new IllegalArgumentException("GGML dequantization input is too short: requires "
                    + requiredInputBytes + " bytes but got " + rawBytes.length());
        }

        int nativeType = nativeQuantType(type);
        if (nativeType < 0) {
            throw new IllegalArgumentException("No native GGML dequantizer is available for " + type);
        }
        GGMLDequantize op = new GGMLDequantize(rawBytes, nativeType, output.dataType(), shape);
        op.addOutputArgument(output);
        Nd4j.exec(op);
    }

    private static int nativeQuantType(GGMLDataType type) {
        // Q5_0 and Q5_1: skip native op path because the GPU kernel currently falls to a
        // zero-fill branch for these types. Use the correct Java dequantizer directly.
        // When the native binary is rebuilt with the fixed CUDA kernel, remove this guard.
        if (type == GGMLDataType.GGML_TYPE_Q5_0 || type == GGMLDataType.GGML_TYPE_Q5_1) {
            return -1;
        }
        return mapToNativeQuantType(type);
    }

    private static DataType directNativeTargetType(DataType targetType) {
        switch (targetType) {
            case FLOAT:
            case HALF:
            case BFLOAT16:
                return targetType;
            default:
                return DataType.FLOAT;
        }
    }

    /**
     * Map GGMLDataType to native ggml_dequantize op quant type integer.
     * Returns -1 if the type is not supported by the native op.
     */
    private static int mapToNativeQuantType(GGMLDataType type) {
        switch (type) {
            case GGML_TYPE_Q4_0: return GGMLDequantize.QUANT_Q4_0;
            case GGML_TYPE_Q4_1: return GGMLDequantize.QUANT_Q4_1;
            case GGML_TYPE_Q5_0: return GGMLDequantize.QUANT_Q5_0;
            case GGML_TYPE_Q5_1: return GGMLDequantize.QUANT_Q5_1;
            case GGML_TYPE_Q8_0: return GGMLDequantize.QUANT_Q8_0;
            case GGML_TYPE_Q8_1: return GGMLDequantize.QUANT_Q8_1;
            case GGML_TYPE_Q2_K: return GGMLDequantize.QUANT_Q2_K;
            case GGML_TYPE_Q3_K: return GGMLDequantize.QUANT_Q3_K;
            case GGML_TYPE_Q4_K: return GGMLDequantize.QUANT_Q4_K;
            case GGML_TYPE_Q5_K: return GGMLDequantize.QUANT_Q5_K;
            case GGML_TYPE_Q6_K: return GGMLDequantize.QUANT_Q6_K;
            case GGML_TYPE_Q8_K: return GGMLDequantize.QUANT_Q8_K;
            case GGML_TYPE_IQ2_XXS: return GGMLDequantize.QUANT_IQ2_XXS;
            case GGML_TYPE_IQ2_XS: return GGMLDequantize.QUANT_IQ2_XS;
            case GGML_TYPE_IQ3_XXS: return GGMLDequantize.QUANT_IQ3_XXS;
            case GGML_TYPE_IQ1_S: return GGMLDequantize.QUANT_IQ1_S;
            case GGML_TYPE_IQ4_NL: return GGMLDequantize.QUANT_IQ4_NL;
            case GGML_TYPE_IQ3_S: return GGMLDequantize.QUANT_IQ3_S;
            case GGML_TYPE_IQ2_S: return GGMLDequantize.QUANT_IQ2_S;
            case GGML_TYPE_IQ4_XS: return GGMLDequantize.QUANT_IQ4_XS;
            case GGML_TYPE_IQ1_M: return GGMLDequantize.QUANT_IQ1_M;
            case GGML_TYPE_TQ1_0: return GGMLDequantize.QUANT_TQ1_0;
            case GGML_TYPE_TQ2_0: return GGMLDequantize.QUANT_TQ2_0;
            default: return -1;
        }
    }

    /**
     * Extract quantization info from data
     */
    public static QuantizationInfo extractQuantizationInfo(byte[] data, GGMLDataType type, long[] shape) {
        return getDequantizer(type).extractQuantizationInfo(data, shape);
    }
}
