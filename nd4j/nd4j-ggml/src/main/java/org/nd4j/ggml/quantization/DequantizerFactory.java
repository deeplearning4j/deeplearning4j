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

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Factory for creating dequantizers based on quantization type.
 */
public class DequantizerFactory {

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
     * Dequantize data to INDArray
     */
    public static INDArray dequantizeToArray(byte[] data, GGMLDataType type, long[] shape, DataType targetType) {
        return getDequantizer(type).dequantizeToArray(data, shape, targetType);
    }

    /**
     * Extract quantization info from data
     */
    public static QuantizationInfo extractQuantizationInfo(byte[] data, GGMLDataType type, long[] shape) {
        return getDequantizer(type).extractQuantizationInfo(data, shape);
    }
}
