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
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Factory for creating quantizers based on target quantization type.
 * This is the reverse of {@link DequantizerFactory}.
 */
public class QuantizerFactory {

    private static final Map<GGMLDataType, Quantizer> quantizers = new ConcurrentHashMap<>();

    static {
        // Register built-in quantizers
        register(new Q8_0Quantizer());
        register(new Q4_0Quantizer());
        register(new Q4_1Quantizer());
        register(new Q5_0Quantizer());
        register(new Q5_1Quantizer());
        register(new Q4_KQuantizer());
        register(new Q5_KQuantizer());
        register(new Q6_KQuantizer());
    }

    private QuantizerFactory() {
        // Utility class
    }

    /**
     * Register a quantizer
     */
    public static void register(Quantizer quantizer) {
        quantizers.put(quantizer.getQuantType(), quantizer);
    }

    /**
     * Get a quantizer for the given type
     *
     * @throws IllegalArgumentException if no quantizer is available
     */
    public static Quantizer getQuantizer(GGMLDataType type) {
        Quantizer quantizer = quantizers.get(type);
        if (quantizer == null) {
            throw new IllegalArgumentException("No quantizer available for type: " + type);
        }
        return quantizer;
    }

    /**
     * Check if a quantizer is available for the given type
     */
    public static boolean hasQuantizer(GGMLDataType type) {
        return quantizers.containsKey(type);
    }

    /**
     * Get all registered quantization types
     */
    public static Set<GGMLDataType> getSupportedTypes() {
        return quantizers.keySet();
    }

    /**
     * Quantize float data to bytes
     */
    public static byte[] quantize(float[] data, GGMLDataType type) {
        return getQuantizer(type).quantize(data);
    }

    /**
     * Quantize an INDArray to bytes
     */
    public static byte[] quantize(INDArray array, GGMLDataType type) {
        return getQuantizer(type).quantize(array);
    }

    /**
     * Get quantization statistics for the given data
     */
    public static Quantizer.QuantizationStats getQuantizationStats(float[] data, GGMLDataType type) {
        return getQuantizer(type).getQuantizationStats(data);
    }

    /**
     * Calculate the output size in bytes for the given number of elements and type
     */
    public static long calculateOutputBytes(long numElements, GGMLDataType type) {
        return getQuantizer(type).calculateOutputBytes(numElements);
    }
}
