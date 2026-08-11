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
import org.nd4j.common.config.ND4JInferenceWeightDataType;
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
     * Whether to build for training (marks weight variables as trainable in SameDiff).
     *
     * <p>This field is independent of {@link #quantizationMode}: setting
     * {@code forTraining=true} with {@code RUNTIME_QUANTIZED_MATMUL} is valid and is
     * the recommended configuration for QLoRA (see {@link #forTrainingQuantized()}).
     * Setting {@code forTraining=true} with any {@code DEQUANTIZE_*} mode is only
     * appropriate for full-parameter fine-tuning where you accept the memory blowup
     * from dequantizing all weights to dense FP32 or FP16.</p>
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
     * KV cache quantization format for the graph builder.
     *
     * <p>When {@code > 0}, the architecture builder creates INT8 (or FP8/INT4) KV cache
     * placeholders instead of float ones and inserts {@code kv_cache_dequantize} nodes
     * between the cache placeholders and {@code DotProductAttentionV2}. This enables
     * compact in-memory storage of KV histories with on-the-fly dequantization during
     * decode — the dequant op is captured into the CUDA graph for zero-overhead replay.</p>
     *
     * <p>Values correspond to {@code KVQuantFormat} in the C++ enum:</p>
     * <ul>
     *   <li>{@code 0} — disabled (default); float KV cache, standard graph</li>
     *   <li>{@code 1} — INT8 symmetric per-row absmax quantization (recommended)</li>
     *   <li>{@code 2} — FP8 E4M3 (requires hardware FP8 support; falls back to INT8 on CPU)</li>
     *   <li>{@code 3} — FP8 E5M2</li>
     *   <li>{@code 4} — INT4 packed (2 values/byte; 4x compression vs float)</li>
     * </ul>
     *
     * <p>Pairing with {@link org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheStrategy#QUANTIZED}
     * is required; setting this without {@code QUANTIZED} strategy has no effect at runtime.</p>
     */
    @Builder.Default
    private int kvQuantFormat = 0;

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
         * Convert quantized weights to FP8 E4M3
         */
        DEQUANTIZE_TO_FLOAT8_E4M3,

        /**
         * Convert quantized weights to FP8 E5M2
         */
        DEQUANTIZE_TO_FLOAT8_E5M2,

        /**
         * Preserve quantization metadata in SDZ for later reconstruction
         */
        PRESERVE_QUANTIZATION,

        /**
         * Hybrid: dequantize some layers (attention), preserve others (FFN)
         */
        HYBRID,

        /**
         * Keep supported quantization types (Q8_0, Q4_K, Q6_K) as packed bytes
         * and insert ggml_qmatmul ops at every linear layer that uses them.
         * Incompatible linear weights are rejected rather than dequantized.
         */
        RUNTIME_QUANTIZED_MATMUL,

        /**
         * Require Q8_0 packed linear weights and runtime quantized matmul.
         */
        RUNTIME_QUANTIZED_INT8,

        /**
         * Require the Q4_K family of packed linear weights and runtime quantized matmul.
         * Q6_K tensors are accepted because GGUF Q4_K_M profiles use them for selected layers.
         */
        RUNTIME_QUANTIZED_INT4
    }

    /**
     * Create default options for inference. FP16 is the default weight storage
     * dtype; {@code nd4j.optimizer.weightDtype} can select a different policy.
     */
    public static ConversionOptions forInference() {
        return forInference(ND4JInferenceWeightDataType.resolve());
    }

    /**
     * Create inference options for an explicit weight storage policy.
     *
     * <p>INT8 and INT4 select strict packed GGUF execution. They do not cast dense
     * weights to integer arrays and they do not silently dequantize incompatible
     * linear weights.</p>
     */
    public static ConversionOptions forInference(ND4JInferenceWeightDataType weightType) {
        ConversionOptionsBuilder builder = ConversionOptions.builder().forTraining(false);
        switch (weightType) {
            case FLOAT32:
                return builder.quantizationMode(QuantizationMode.DEQUANTIZE_TO_FLOAT32)
                        .targetDataType(DataType.FLOAT).build();
            case FLOAT16:
                return builder.quantizationMode(QuantizationMode.DEQUANTIZE_TO_FLOAT16)
                        .targetDataType(DataType.HALF).build();
            case BFLOAT16:
                return builder.quantizationMode(QuantizationMode.DEQUANTIZE_TO_BFLOAT16)
                        .targetDataType(DataType.BFLOAT16).build();
            case FLOAT8_E4M3:
                return builder.quantizationMode(QuantizationMode.DEQUANTIZE_TO_FLOAT8_E4M3)
                        .targetDataType(DataType.FLOAT8).build();
            case FLOAT8_E5M2:
                return builder.quantizationMode(QuantizationMode.DEQUANTIZE_TO_FLOAT8_E5M2)
                        .targetDataType(DataType.FLOAT8_E5M2).build();
            case INT8:
                return builder.quantizationMode(QuantizationMode.RUNTIME_QUANTIZED_INT8)
                        .targetDataType(DataType.HALF).build();
            case INT4:
                return builder.quantizationMode(QuantizationMode.RUNTIME_QUANTIZED_INT4)
                        .targetDataType(DataType.HALF).build();
            default:
                throw new IllegalArgumentException("Unsupported inference weight dtype: " + weightType);
        }
    }

    /**
     * Create options for full-parameter fine-tuning.
     *
     * <p><b>WARNING — memory blowup:</b> this preset dequantizes ALL weights to dense FP32.
     * A 13 GB Q8_0 GGUF becomes approximately 50 GB of FP32 tensors.  Only use this preset
     * when you intend to train every parameter from scratch or when quantization is irrelevant
     * (e.g. a non-quantized GGUF model).</p>
     *
     * <p>For LoRA / QLoRA on a quantized GGUF base model, use {@link #forTrainingQuantized()}
     * instead: it keeps supported weight types (Q8_0, Q4_K, Q6_K) packed as INT8 bytes and
     * emits {@code ggml_qmatmul} ops, so a 13 GB GGUF stays 13 GB while the LoRA adapters
     * added by {@code PeftModel} are the only new trainable parameters.</p>
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

    /**
     * Create options for runtime quantized matmul.
     * Supported types (Q8_0, Q4_K, Q6_K) are kept as raw packed bytes;
     * ggml_qmatmul ops are wired in place of normal matmuls for those weights.
     * Incompatible linear weights fail conversion instead of falling back.
     */
    public static ConversionOptions runtimeQuantizedMatmul() {
        return ConversionOptions.builder()
                .quantizationMode(QuantizationMode.RUNTIME_QUANTIZED_MATMUL)
                .targetDataType(DataType.HALF)
                .build();
    }

    /**
     * Create options for QLoRA (Quantized Low-Rank Adaptation) on a GGUF base model.
     *
     * <p>This is the correct preset for parameter-efficient fine-tuning (LoRA/QLoRA) of a
     * quantized GGUF model:</p>
     * <ul>
     *   <li>Base weights whose GGUF quantization type is supported (Q8_0, Q4_K, Q6_K) remain
     *       packed as raw INT8 bytes; the graph emits {@code ggml_qmatmul} ops for them.
     *       A 13 GB Q8_0 GGUF stays roughly 13 GB on disk and in memory — NOT 50 GB of
     *       dense FP32 shards.</li>
     *   <li>Unsupported quantized linear weights fail conversion instead of creating a mixed graph.</li>
     *   <li>{@code forTraining=true} marks weight variables as trainable in SameDiff so that
     *       a {@code PeftModel} can freeze the base weights and attach LoRA adapter parameters.</li>
     *   <li>Working dtype is FP16 so activations and LoRA adapters share a compact dtype.</li>
     * </ul>
     *
     * <p>The base weights registered as {@code sd.var} with the GGUF tensor name (e.g.
     * {@code "blk.0.attn_q.weight"}) allow a {@code PeftModel} with
     * {@code targetModules = "attn_q"} to locate and freeze/attach LoRA to them.
     * {@code PeftModel} reads N, K, and quantType from the {@code ggml_qmatmul} op's
     * {@code iArgs} — no additional metadata is required.</p>
     *
     * <p><b>Do NOT use {@link #forTraining()} for LoRA/QLoRA.</b>
     * {@code forTraining()} dequantizes everything to FP32 (a 13 GB GGUF becomes ~50 GB of
     * dense shards) and is only appropriate for full-parameter fine-tuning where every weight
     * must be a dense FP32 gradient target.</p>
     *
     * @return conversion options suitable for QLoRA base model loading
     */
    public static ConversionOptions forTrainingQuantized() {
        return ConversionOptions.builder()
                .quantizationMode(QuantizationMode.RUNTIME_QUANTIZED_MATMUL)
                .targetDataType(DataType.HALF)
                .forTraining(true)
                .build();
    }

    /**
     * Returns true if this ConversionOptions is configured for runtime quantized matmul.
     */
    public boolean isRuntimeQuantizedMatmul() {
        return quantizationMode == QuantizationMode.RUNTIME_QUANTIZED_MATMUL
                || quantizationMode == QuantizationMode.RUNTIME_QUANTIZED_INT8
                || quantizationMode == QuantizationMode.RUNTIME_QUANTIZED_INT4;
    }

    /**
     * Returns {@code true} when the KV cache should be built with INT8/FP8 quantized placeholders
     * and in-graph {@code kv_cache_dequantize} nodes.
     */
    public boolean isKvCacheQuantized() {
        return kvQuantFormat > 0;
    }

    /**
     * Create options with INT8-quantized KV cache placeholders (4x memory vs float).
     * Weights are dequantized to FP16; KV cache uses INT8 per-row absmax quantization.
     */
    public static ConversionOptions withInt8KvCache() {
        return ConversionOptions.builder()
                .quantizationMode(QuantizationMode.DEQUANTIZE_TO_FLOAT16)
                .targetDataType(DataType.HALF)
                .kvQuantFormat(1) // INT8
                .build();
    }
}
