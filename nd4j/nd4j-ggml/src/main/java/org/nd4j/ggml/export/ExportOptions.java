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

package org.nd4j.ggml.export;

import lombok.Builder;
import lombok.Data;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.ggml.format.GGMLFormat;

import java.util.Map;

/**
 * Configuration options for SameDiff to GGML/GGUF export.
 */
@Data
@Builder
public class ExportOptions {

    /**
     * How to quantize weights during export
     */
    public enum QuantizationType {
        /** No quantization, full 32-bit precision */
        F32(GGMLDataType.GGML_TYPE_F32),
        /** Half precision (FP16) */
        F16(GGMLDataType.GGML_TYPE_F16),
        /** BFloat16 */
        BF16(GGMLDataType.GGML_TYPE_BF16),
        /** 4-bit quantization (simple) */
        Q4_0(GGMLDataType.GGML_TYPE_Q4_0),
        /** 4-bit with min values */
        Q4_1(GGMLDataType.GGML_TYPE_Q4_1),
        /** 5-bit quantization */
        Q5_0(GGMLDataType.GGML_TYPE_Q5_0),
        /** 5-bit with min values */
        Q5_1(GGMLDataType.GGML_TYPE_Q5_1),
        /** 8-bit quantization */
        Q8_0(GGMLDataType.GGML_TYPE_Q8_0),
        /** 4-bit k-quant (recommended for quality/size balance) */
        Q4_K(GGMLDataType.GGML_TYPE_Q4_K),
        /** 5-bit k-quant */
        Q5_K(GGMLDataType.GGML_TYPE_Q5_K),
        /** 6-bit k-quant (highest quality quantization) */
        Q6_K(GGMLDataType.GGML_TYPE_Q6_K);

        private final GGMLDataType ggmlDataType;

        QuantizationType(GGMLDataType ggmlDataType) {
            this.ggmlDataType = ggmlDataType;
        }

        public GGMLDataType toGGMLDataType() {
            return ggmlDataType;
        }
    }

    /**
     * Default quantization type for export
     */
    @Builder.Default
    private QuantizationType quantizationType = QuantizationType.F16;

    /**
     * Output format (GGUF recommended, GGML for legacy compatibility)
     */
    @Builder.Default
    private GGMLFormat outputFormat = GGMLFormat.GGUF;

    /**
     * GGUF version to write (1, 2, or 3)
     */
    @Builder.Default
    private int ggufVersion = 3;

    /**
     * Alignment for tensor data section (default 32 bytes)
     */
    @Builder.Default
    private int alignment = 32;

    /**
     * Override architecture detection (null = auto-detect)
     */
    private String architectureHint;

    /**
     * Model name for GGUF metadata
     */
    private String modelName;

    /**
     * Model author for GGUF metadata
     */
    private String modelAuthor;

    /**
     * Model description for GGUF metadata
     */
    private String modelDescription;

    /**
     * Whether to include tokenizer information in export
     */
    @Builder.Default
    private boolean includeTokenizer = true;

    /**
     * Per-layer quantization overrides (tensor name pattern -> quantization type)
     * Allows different quantization for different layers (e.g., keep embeddings in FP16)
     */
    private Map<String, QuantizationType> layerQuantizationOverrides;

    /**
     * Custom tensor name mapping (SameDiff name -> GGML name)
     */
    private Map<String, String> tensorNameMapping;

    /**
     * Whether to validate the model before export
     */
    @Builder.Default
    private boolean validateBeforeExport = true;

    /**
     * Whether to include model statistics in metadata
     */
    @Builder.Default
    private boolean includeStatistics = false;

    // ========== Factory Methods ==========

    /**
     * Create default options for FP16 export (no quantization)
     */
    public static ExportOptions f16() {
        return ExportOptions.builder()
                .quantizationType(QuantizationType.F16)
                .build();
    }

    /**
     * Create default options for FP32 export (full precision)
     */
    public static ExportOptions f32() {
        return ExportOptions.builder()
                .quantizationType(QuantizationType.F32)
                .build();
    }

    /**
     * Create options for Q4_K quantization (recommended balance of quality/size)
     */
    public static ExportOptions q4k() {
        return ExportOptions.builder()
                .quantizationType(QuantizationType.Q4_K)
                .build();
    }

    /**
     * Create options for Q8_0 quantization (higher quality, larger size)
     */
    public static ExportOptions q8_0() {
        return ExportOptions.builder()
                .quantizationType(QuantizationType.Q8_0)
                .build();
    }

    /**
     * Create options for Q5_K quantization (good balance)
     */
    public static ExportOptions q5k() {
        return ExportOptions.builder()
                .quantizationType(QuantizationType.Q5_K)
                .build();
    }

    /**
     * Create options for Q6_K quantization (highest quality quantization)
     */
    public static ExportOptions q6k() {
        return ExportOptions.builder()
                .quantizationType(QuantizationType.Q6_K)
                .build();
    }

    /**
     * Create options for legacy GGML format export
     */
    public static ExportOptions legacyGGML() {
        return ExportOptions.builder()
                .outputFormat(GGMLFormat.GGML)
                .quantizationType(QuantizationType.F16)
                .build();
    }
}
