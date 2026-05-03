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

package org.nd4j.ggml.convert;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

/**
 * Configuration options for GGML to SDZ conversion.
 */
@Data
@Builder
public class ConversionOptions {

    /**
     * How to handle quantized weights
     */
    @Builder.Default
    private QuantizationMode quantizationMode = QuantizationMode.DEQUANTIZE_TO_FLOAT16;

    /**
     * Target data type for dequantization
     */
    @Builder.Default
    private DataType targetDataType = DataType.FLOAT16;

    /**
     * Whether to preserve tokenizer information in metadata
     */
    @Builder.Default
    private boolean preserveTokenizerInfo = true;

    /**
     * Whether to build for training (includes gradient support)
     */
    @Builder.Default
    private boolean forTraining = false;

    /**
     * Whether to use memory mapping for large files
     */
    @Builder.Default
    private boolean useMemoryMapping = true;

    /**
     * Batch size for tensor conversion (memory management)
     */
    @Builder.Default
    private int tensorBatchSize = 10;

    /**
     * Override architecture detection (null = auto-detect)
     */
    private String architectureOverride;

    /**
     * Custom tensor name mapping (optional)
     */
    private Map<String, String> tensorNameMapping;

    /**
     * Maximum file size to process (bytes, 0 = unlimited)
     */
    @Builder.Default
    private long maxFileSize = 0;

    /**
     * How to handle quantized weights during conversion
     */
    public enum QuantizationMode {
        /**
         * Convert quantized weights to FP32
         */
        DEQUANTIZE_TO_FLOAT32,

        /**
         * Convert quantized weights to FP16
         */
        DEQUANTIZE_TO_FLOAT16,

        /**
         * Convert quantized weights to BF16
         */
        DEQUANTIZE_TO_BFLOAT16,

        /**
         * Preserve quantization metadata in SDZ for later reconstruction
         */
        PRESERVE_QUANTIZATION,

        /**
         * Hybrid: dequantize some layers (attention), preserve others (FFN)
         */
        HYBRID
    }

    /**
     * Create default options for inference.
     * Uses FP32 on all backends. FP16 weights with explicit cast ops cause
     * cumulative precision loss over many transformer layers (669 HALF↔FLOAT32
     * round-trips on a 24-layer model produce wrong logits). cuBLAS and
     * rms_norm already use FP32 accumulation internally, so FP32 weights
     * avoid unnecessary casts while maintaining the same compute precision.
     */
    public static ConversionOptions forInference() {
        return ConversionOptions.builder()
                .forTraining(false)
                .quantizationMode(QuantizationMode.DEQUANTIZE_TO_FLOAT32)
                .targetDataType(DataType.FLOAT)
                .build();
    }

    /**
     * Create default options for training/fine-tuning
     */
    public static ConversionOptions forTraining() {
        return ConversionOptions.builder()
                .forTraining(true)
                .quantizationMode(QuantizationMode.DEQUANTIZE_TO_FLOAT32)
                .targetDataType(DataType.FLOAT)
                .build();
    }

    /**
     * Create options with FP16 precision
     */
    public static ConversionOptions fp16() {
        return ConversionOptions.builder()
                .quantizationMode(QuantizationMode.DEQUANTIZE_TO_FLOAT16)
                .targetDataType(DataType.HALF)
                .build();
    }

    /**
     * Create options that preserve quantization
     */
    public static ConversionOptions preserveQuantization() {
        return ConversionOptions.builder()
                .quantizationMode(QuantizationMode.PRESERVE_QUANTIZATION)
                .build();
    }
}
