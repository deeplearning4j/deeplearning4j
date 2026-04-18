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

import org.nd4j.ggml.export.ExportOptions;
import org.nd4j.ggml.format.GGMLDataType;

/**
 * Enumeration of GGML quantization types with their associated bits-per-weight values.
 *
 * <p>This enum bridges the per-layer adaptive quantization system with the underlying
 * {@link GGMLDataType} and {@link ExportOptions.QuantizationType} representations.
 * Bits-per-weight values match the GGML specification and are used to estimate
 * model size when assigning quantization budgets.
 *
 * <p>Ordering: entries are ordered from most aggressive (fewest bits) to highest
 * quality (most bits), which matches the assignment logic in
 * {@link AdaptiveLayerQuantizer}.
 */
public enum GGMLQuantType {

    /** 2-bit k-quant (~2.56 bpw). Very aggressive, noticeable quality loss. */
    Q2_K(2.5625, GGMLDataType.GGML_TYPE_Q2_K, ExportOptions.QuantizationType.Q4_0),

    /** 3-bit k-quant (~3.44 bpw). Aggressive quantization, reasonable for middle layers. */
    Q3_K(3.4375, GGMLDataType.GGML_TYPE_Q3_K, null),

    /** 4-bit simple quantization (~4.5 bpw). */
    Q4_0(4.5, GGMLDataType.GGML_TYPE_Q4_0, ExportOptions.QuantizationType.Q4_0),

    /** 4-bit with minimum value (~4.625 bpw). */
    Q4_1(4.625, GGMLDataType.GGML_TYPE_Q4_1, ExportOptions.QuantizationType.Q4_1),

    /** 4-bit k-quant (~4.5 bpw, better quality than Q4_0). Recommended default. */
    Q4_K(4.5, GGMLDataType.GGML_TYPE_Q4_K, ExportOptions.QuantizationType.Q4_K),

    /** 5-bit simple quantization (~5.5 bpw). */
    Q5_0(5.5, GGMLDataType.GGML_TYPE_Q5_0, ExportOptions.QuantizationType.Q5_0),

    /** 5-bit with minimum value (~5.625 bpw). */
    Q5_1(5.625, GGMLDataType.GGML_TYPE_Q5_1, ExportOptions.QuantizationType.Q5_1),

    /** 5-bit k-quant (~5.5 bpw, better quality than Q5_0). */
    Q5_K(5.5, GGMLDataType.GGML_TYPE_Q5_K, ExportOptions.QuantizationType.Q5_K),

    /** 6-bit k-quant (~6.5625 bpw). Highest quality quantized format. */
    Q6_K(6.5625, GGMLDataType.GGML_TYPE_Q6_K, ExportOptions.QuantizationType.Q6_K),

    /** 8-bit simple quantization (~8.5 bpw). Near-lossless for most weights. */
    Q8_0(8.5, GGMLDataType.GGML_TYPE_Q8_0, ExportOptions.QuantizationType.Q8_0),

    /** 16-bit half precision (16 bpw). Full FP16 - no quantization loss. */
    F16(16.0, GGMLDataType.GGML_TYPE_F16, ExportOptions.QuantizationType.F16),

    /** 32-bit full precision (32 bpw). Reference quality. */
    F32(32.0, GGMLDataType.GGML_TYPE_F32, ExportOptions.QuantizationType.F32);

    private final double bitsPerWeight;
    private final GGMLDataType ggmlDataType;
    private final ExportOptions.QuantizationType exportQuantizationType;

    GGMLQuantType(double bitsPerWeight, GGMLDataType ggmlDataType,
                  ExportOptions.QuantizationType exportQuantizationType) {
        this.bitsPerWeight = bitsPerWeight;
        this.ggmlDataType = ggmlDataType;
        this.exportQuantizationType = exportQuantizationType;
    }

    /**
     * Average bits per weight for this quantization type.
     * Used for model size estimation when assigning per-layer budgets.
     */
    public double getBitsPerWeight() {
        return bitsPerWeight;
    }

    /**
     * The underlying GGML data type constant.
     */
    public GGMLDataType toGGMLDataType() {
        return ggmlDataType;
    }

    /**
     * The corresponding {@link ExportOptions.QuantizationType}, or {@code null} if
     * there is no direct mapping (e.g. Q3_K which has no ExportOptions entry yet).
     */
    public ExportOptions.QuantizationType toExportQuantizationType() {
        return exportQuantizationType;
    }

    /**
     * Estimate the number of bytes required to store {@code numElements} weights
     * at this quantization type's precision.
     *
     * @param numElements number of scalar weight values
     * @return estimated byte count
     */
    public long estimateBytes(long numElements) {
        return (long) Math.ceil(numElements * bitsPerWeight / 8.0);
    }

    /**
     * Look up a {@code GGMLQuantType} by the underlying {@link GGMLDataType}.
     *
     * @param ggmlDataType the GGML data type to find
     * @return matching enum constant
     * @throws IllegalArgumentException if no matching type exists
     */
    public static GGMLQuantType fromGGMLDataType(GGMLDataType ggmlDataType) {
        for (GGMLQuantType qt : values()) {
            if (qt.ggmlDataType == ggmlDataType) {
                return qt;
            }
        }
        throw new IllegalArgumentException("No GGMLQuantType for GGMLDataType: " + ggmlDataType);
    }

    /**
     * Look up a {@code GGMLQuantType} by the {@link ExportOptions.QuantizationType}.
     *
     * @param exportType the export quantization type
     * @return matching enum constant
     * @throws IllegalArgumentException if no matching type exists
     */
    public static GGMLQuantType fromExportQuantizationType(ExportOptions.QuantizationType exportType) {
        for (GGMLQuantType qt : values()) {
            if (qt.exportQuantizationType == exportType) {
                return qt;
            }
        }
        throw new IllegalArgumentException("No GGMLQuantType for ExportQuantizationType: " + exportType);
    }
}
