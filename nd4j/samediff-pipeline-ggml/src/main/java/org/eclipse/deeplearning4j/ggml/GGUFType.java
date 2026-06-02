/*
 *  ******************************************************************************
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

package org.eclipse.deeplearning4j.ggml;

import org.nd4j.linalg.api.buffer.DataType;

/**
 * GGUF/GGML tensor data types.
 * See: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 */
public enum GGUFType {
    F32(0, "F32", DataType.FLOAT, 4, false),
    F16(1, "F16", DataType.HALF, 2, false),
    Q4_0(2, "Q4_0", DataType.INT8, 0, true),
    Q4_1(3, "Q4_1", DataType.INT8, 0, true),
    Q5_0(6, "Q5_0", DataType.INT8, 0, true),
    Q5_1(7, "Q5_1", DataType.INT8, 0, true),
    Q8_0(8, "Q8_0", DataType.INT8, 0, true),
    Q8_1(9, "Q8_1", DataType.INT8, 0, true),
    Q2_K(10, "Q2_K", DataType.INT8, 0, true),
    Q3_K(11, "Q3_K", DataType.INT8, 0, true),
    Q4_K(12, "Q4_K", DataType.INT8, 0, true),
    Q5_K(13, "Q5_K", DataType.INT8, 0, true),
    Q6_K(14, "Q6_K", DataType.INT8, 0, true),
    Q8_K(15, "Q8_K", DataType.INT8, 0, true),
    IQ2_XXS(16, "IQ2_XXS", DataType.INT8, 0, true),
    IQ2_XS(17, "IQ2_XS", DataType.INT8, 0, true),
    IQ3_XXS(18, "IQ3_XXS", DataType.INT8, 0, true),
    IQ1_S(19, "IQ1_S", DataType.INT8, 0, true),
    IQ4_NL(20, "IQ4_NL", DataType.INT8, 0, true),
    IQ3_S(21, "IQ3_S", DataType.INT8, 0, true),
    IQ2_S(22, "IQ2_S", DataType.INT8, 0, true),
    IQ4_XS(23, "IQ4_XS", DataType.INT8, 0, true),
    I8(24, "I8", DataType.INT8, 1, false),
    I16(25, "I16", DataType.INT16, 2, false),
    I32(26, "I32", DataType.INT32, 4, false),
    I64(27, "I64", DataType.INT64, 8, false),
    F64(28, "F64", DataType.DOUBLE, 8, false),
    BF16(30, "BF16", DataType.BFLOAT16, 2, false);

    private final int id;
    private final String name;
    private final DataType nd4jType;
    private final int bytesPerElement;
    private final boolean quantized;

    GGUFType(int id, String name, DataType nd4jType, int bytesPerElement, boolean quantized) {
        this.id = id;
        this.name = name;
        this.nd4jType = nd4jType;
        this.bytesPerElement = bytesPerElement;
        this.quantized = quantized;
    }

    public int getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public DataType toNd4jType() {
        return nd4jType;
    }

    public int getBytesPerElement() {
        return bytesPerElement;
    }

    public boolean isQuantized() {
        return quantized;
    }

    public static GGUFType fromId(int id) {
        for (GGUFType type : values()) {
            if (type.id == id) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown GGUF type id: " + id);
    }

    public static GGUFType fromName(String name) {
        for (GGUFType type : values()) {
            if (type.name.equalsIgnoreCase(name)) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown GGUF type name: " + name);
    }

    public long calculateBlockSize() {
        switch (this) {
            case Q4_0:
                return 18;
            case Q4_1:
                return 20;
            case Q5_0:
                return 22;
            case Q5_1:
                return 24;
            case Q8_0:
                return 34;
            case Q8_1:
                return 36;
            case Q2_K:
                return 84;
            case Q3_K:
                return 110;
            case Q4_K:
                return 144;
            case Q5_K:
                return 176;
            case Q6_K:
                return 210;
            case Q8_K:
                return 292;
            default:
                return bytesPerElement;
        }
    }

    public int getBlockElements() {
        if (!quantized) return 1;
        switch (this) {
            case Q4_0:
            case Q4_1:
            case Q5_0:
            case Q5_1:
            case Q8_0:
            case Q8_1:
                return 32;
            case Q2_K:
            case Q3_K:
            case Q4_K:
            case Q5_K:
            case Q6_K:
            case Q8_K:
                return 256;
            default:
                return 32;
        }
    }
}
