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

package org.nd4j.ggml.format;

import org.nd4j.linalg.api.buffer.DataType;

/**
 * GGML data types enumeration.
 * Maps GGML tensor types to their properties and ND4J equivalents where applicable.
 *
 * Based on GGML source code: https://github.com/ggerganov/ggml
 */
public enum GGMLDataType {
    // Full precision types
    GGML_TYPE_F32(0, DataType.FLOAT, 4.0, 1, false),
    GGML_TYPE_F16(1, DataType.HALF, 2.0, 1, false),

    // Quantized types - legacy
    GGML_TYPE_Q4_0(2, null, 0.5 + 2.0/32, 32, true),    // 4-bit, block=32, 18 bytes/block
    GGML_TYPE_Q4_1(3, null, 0.5 + 4.0/32, 32, true),    // 4-bit with min, block=32, 20 bytes/block
    GGML_TYPE_Q5_0(6, null, 0.5 + 2.0/32 + 4.0/32, 32, true),  // 5-bit
    GGML_TYPE_Q5_1(7, null, 0.5 + 4.0/32 + 4.0/32, 32, true),  // 5-bit with min
    GGML_TYPE_Q8_0(8, null, 1.0 + 2.0/32, 32, true),    // 8-bit, block=32, 34 bytes/block
    GGML_TYPE_Q8_1(9, null, 1.0 + 4.0/32, 32, true),    // 8-bit with sum

    // K-quant types (improved quantization)
    GGML_TYPE_Q2_K(10, null, 2.5625/8, 256, true),      // 2-bit k-quant
    GGML_TYPE_Q3_K(11, null, 3.4375/8, 256, true),      // 3-bit k-quant
    GGML_TYPE_Q4_K(12, null, 4.5/8, 256, true),         // 4-bit k-quant
    GGML_TYPE_Q5_K(13, null, 5.5/8, 256, true),         // 5-bit k-quant
    GGML_TYPE_Q6_K(14, null, 6.5625/8, 256, true),      // 6-bit k-quant
    GGML_TYPE_Q8_K(15, null, 8.0/8 + 2.0/256, 256, true), // 8-bit k-quant

    // Importance quantization types
    GGML_TYPE_IQ2_XXS(16, null, 2.0625/8, 256, true),
    GGML_TYPE_IQ2_XS(17, null, 2.3125/8, 256, true),
    GGML_TYPE_IQ3_XXS(18, null, 3.0625/8, 256, true),
    GGML_TYPE_IQ1_S(19, null, 1.5625/8, 256, true),
    GGML_TYPE_IQ4_NL(20, null, 4.5/8, 256, true),
    GGML_TYPE_IQ3_S(21, null, 3.4375/8, 256, true),
    GGML_TYPE_IQ2_S(22, null, 2.5/8, 256, true),
    GGML_TYPE_IQ4_XS(23, null, 4.25/8, 256, true),
    GGML_TYPE_IQ1_M(24, null, 1.75/8, 256, true),

    // Integer types
    GGML_TYPE_I8(25, DataType.BYTE, 1.0, 1, false),
    GGML_TYPE_I16(26, DataType.SHORT, 2.0, 1, false),
    GGML_TYPE_I32(27, DataType.INT, 4.0, 1, false),
    GGML_TYPE_I64(28, DataType.LONG, 8.0, 1, false),

    // Double precision
    GGML_TYPE_F64(29, DataType.DOUBLE, 8.0, 1, false),

    // BFloat16
    GGML_TYPE_BF16(30, DataType.BFLOAT16, 2.0, 1, false),

    // Ternary quantization
    GGML_TYPE_TQ1_0(34, null, 1.6875/8, 256, true),
    GGML_TYPE_TQ2_0(35, null, 2.0625/8, 256, true);

    private final int typeId;
    private final DataType nd4jType;
    private final double bytesPerElement;
    private final int blockSize;
    private final boolean quantized;

    GGMLDataType(int typeId, DataType nd4jType, double bytesPerElement, int blockSize, boolean quantized) {
        this.typeId = typeId;
        this.nd4jType = nd4jType;
        this.bytesPerElement = bytesPerElement;
        this.blockSize = blockSize;
        this.quantized = quantized;
    }

    /**
     * Get the GGML type ID
     */
    public int getTypeId() {
        return typeId;
    }

    /**
     * Get the corresponding ND4J DataType, or null for quantized types
     */
    public DataType getNd4jType() {
        return nd4jType;
    }

    /**
     * Get bytes per element (average for quantized types)
     */
    public double getBytesPerElement() {
        return bytesPerElement;
    }

    /**
     * Get the block size for quantization (1 for non-quantized types)
     */
    public int getBlockSize() {
        return blockSize;
    }

    /**
     * Check if this is a quantized type
     */
    public boolean isQuantized() {
        return quantized;
    }

    /**
     * Calculate the number of bytes needed to store a tensor with the given number of elements
     */
    public long calculateStorageBytes(long numElements) {
        if (!quantized) {
            return (long) (numElements * bytesPerElement);
        }
        // For quantized types, round up to full blocks
        long numBlocks = (numElements + blockSize - 1) / blockSize;
        return (long) (numBlocks * blockSize * bytesPerElement);
    }

    /**
     * Lookup a GGMLDataType by its type ID
     */
    public static GGMLDataType fromTypeId(int typeId) {
        for (GGMLDataType type : values()) {
            if (type.typeId == typeId) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown GGML type ID: " + typeId);
    }

    /**
     * Check if a type ID is valid
     */
    public static boolean isValidTypeId(int typeId) {
        for (GGMLDataType type : values()) {
            if (type.typeId == typeId) {
                return true;
            }
        }
        return false;
    }

    /**
     * Convert this GGMLDataType to the C++ GgmlQuantType positional enum index
     * used by ggml_dequantize and ggml_qmatmul.
     * Returns -1 if this type is not a supported quantized type.
     *
     * Note: The C++ GgmlQuantType enum uses positional indices starting at 0,
     * which differ from the GGUF file typeIds stored in this enum.
     */
    public int toGgmlQuantType() {
        switch (this) {
            case GGML_TYPE_Q4_0:  return 0;
            case GGML_TYPE_Q4_1:  return 1;
            case GGML_TYPE_Q5_0:  return 2;
            case GGML_TYPE_Q5_1:  return 3;
            case GGML_TYPE_Q8_0:  return 4;
            case GGML_TYPE_Q8_1:  return 5;
            case GGML_TYPE_Q2_K:  return 6;
            case GGML_TYPE_Q3_K:  return 7;
            case GGML_TYPE_Q4_K:  return 8;
            case GGML_TYPE_Q5_K:  return 9;
            case GGML_TYPE_Q6_K:  return 10;
            case GGML_TYPE_Q8_K:  return 11;
            default: return -1;
        }
    }

    /**
     * Returns true if this type is supported by ggml_qmatmul (v1).
     * Supported: Q8_0, Q4_K, Q6_K.
     */
    public boolean isRuntimeQMatMulSupported() {
        return this == GGML_TYPE_Q8_0 || this == GGML_TYPE_Q4_K || this == GGML_TYPE_Q6_K;
    }
}
