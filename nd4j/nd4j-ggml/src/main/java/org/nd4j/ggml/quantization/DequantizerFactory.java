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
     * Dequantizes directly to the target type (F32, F16, BF16) without
     * intermediate allocations when the native op supports the target type.
     * Falls back to Java dequantizers if the native op fails or the type
     * is not supported.
     */
    public static INDArray dequantizeToArray(byte[] data, GGMLDataType type, long[] shape, DataType targetType) {
        int nativeType = mapToNativeQuantType(type);
        if (nativeType >= 0) {
            try {
                INDArray rawBytes = Nd4j.createFromArray(data).castTo(DataType.INT8);
                GGMLDequantize op = new GGMLDequantize(rawBytes, nativeType, targetType, shape);
                INDArray result = Nd4j.exec(op)[0];
                rawBytes.close();

                // The native dequant op runs on GPU and writes to the device (special) buffer.
                // The host (primary) buffer remains stale/zeros. We must sync device→host
                // so that ModelLoadingContext.scheduleTransfer() (which does host→device)
                // doesn't overwrite the good device data with stale host zeros.
                Nd4j.getAffinityManager().ensureLocation(result,
                    org.nd4j.linalg.api.concurrency.AffinityManager.Location.HOST);

                return result;
            } catch (Exception e) {
                long numElements = 1;
                for (long dim : shape) numElements *= dim;
                if (numElements > Integer.MAX_VALUE) {
                    throw new RuntimeException(
                        "Native ggml_dequantize failed for " + type + " with " + numElements +
                        " elements. Java fallback cannot handle tensors this large. " +
                        "Native error: " + e.getMessage(), e);
                }
                log.debug("Native ggml_dequantize failed for {}, falling back to Java dequantizer: {}",
                          type, e.getMessage());
            }
        }

        // Fallback: Java dequantizer (always produces F32, then cast)
        INDArray fp32 = getDequantizer(type).dequantizeToArray(data, shape, DataType.FLOAT);
        if (targetType == DataType.FLOAT) {
            return fp32;
        }
        INDArray result = fp32.castTo(targetType);
        fp32.close();
        return result;
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
