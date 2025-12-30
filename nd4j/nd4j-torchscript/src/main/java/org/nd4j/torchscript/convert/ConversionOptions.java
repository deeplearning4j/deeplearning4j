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

package org.nd4j.torchscript.convert;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.HashMap;
import java.util.Map;

/**
 * Options for TorchScript to SameDiff conversion.
 */
@Data
@Builder
public class ConversionOptions {

    /**
     * Target data type for converted tensors.
     */
    @Builder.Default
    private DataType targetDataType = DataType.FLOAT;

    /**
     * Whether to preserve training mode (gradients, dropout, etc.).
     */
    @Builder.Default
    private boolean forTraining = false;

    /**
     * Whether to use memory-mapped file access.
     */
    @Builder.Default
    private boolean useMemoryMapping = true;

    /**
     * Override architecture detection with a specific architecture name.
     */
    private String architectureOverride;

    /**
     * Custom tensor name mappings (PyTorch name -> SameDiff name).
     */
    @Builder.Default
    private Map<String, String> tensorNameMapping = new HashMap<>();

    /**
     * Maximum file size to process (0 = unlimited).
     */
    @Builder.Default
    private long maxFileSize = 0;

    /**
     * Whether to preserve original input names from the model.
     */
    @Builder.Default
    private boolean preserveInputNames = true;

    /**
     * Data format for convolutions (NCHW for PyTorch, NHWC for TensorFlow-style).
     */
    @Builder.Default
    private DataFormat dataFormat = DataFormat.NCHW;

    /**
     * Whether to convert weights from PyTorch format to ND4J format.
     */
    @Builder.Default
    private boolean transformWeights = true;

    /**
     * Whether to skip unsupported operations (log warning) or fail.
     */
    @Builder.Default
    private boolean skipUnsupportedOps = false;

    /**
     * Default input shape for models without explicit input shape.
     */
    private long[] defaultInputShape;

    /**
     * Create options optimized for inference.
     *
     * @return inference-optimized options
     */
    public static ConversionOptions forInference() {
        return ConversionOptions.builder()
                .forTraining(false)
                .targetDataType(DataType.FLOAT)
                .build();
    }

    /**
     * Create options optimized for training.
     *
     * @return training-optimized options
     */
    public static ConversionOptions forTraining() {
        return ConversionOptions.builder()
                .forTraining(true)
                .targetDataType(DataType.FLOAT)
                .build();
    }

    /**
     * Create options for FP16 inference.
     *
     * @return FP16 options
     */
    public static ConversionOptions fp16() {
        return ConversionOptions.builder()
                .forTraining(false)
                .targetDataType(DataType.HALF)
                .build();
    }

    /**
     * Create options for BF16 inference.
     *
     * @return BF16 options
     */
    public static ConversionOptions bf16() {
        return ConversionOptions.builder()
                .forTraining(false)
                .targetDataType(DataType.BFLOAT16)
                .build();
    }

    /**
     * Data format enum for convolution operations.
     */
    public enum DataFormat {
        /**
         * PyTorch default: [batch, channels, height, width]
         */
        NCHW,

        /**
         * TensorFlow default: [batch, height, width, channels]
         */
        NHWC
    }
}
