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

package org.nd4j.torchscript.format;

import lombok.Builder;
import lombok.Data;

import java.util.Arrays;

/**
 * Information about a single tensor in a TorchScript or Safetensors model.
 */
@Data
@Builder
public class TorchScriptTensorInfo {

    /**
     * Name of the tensor (e.g., "layer1.0.conv1.weight")
     */
    private String name;

    /**
     * Number of dimensions
     */
    private int numDimensions;

    /**
     * Shape of the tensor
     */
    private long[] shape;

    /**
     * Data type of the tensor
     */
    private TorchScriptDataType dataType;

    /**
     * Offset of tensor data in the file
     */
    private long dataOffset;

    /**
     * Size of tensor data in bytes
     */
    private long dataSize;

    /**
     * Whether this tensor requires gradient (is a trainable parameter)
     */
    @Builder.Default
    private boolean requiresGrad = false;

    /**
     * Whether this is a buffer (non-trainable, e.g., running_mean)
     */
    @Builder.Default
    private boolean isBuffer = false;

    /**
     * Get the total number of elements in this tensor.
     *
     * @return number of elements
     */
    public long getNumElements() {
        if (shape == null || shape.length == 0) {
            return 0;
        }
        long elements = 1;
        for (long dim : shape) {
            elements *= dim;
        }
        return elements;
    }

    /**
     * Get the expected data size based on shape and data type.
     *
     * @return expected size in bytes
     */
    public long getExpectedDataSize() {
        return getNumElements() * dataType.getBytesPerElement();
    }

    /**
     * Get a human-readable shape string.
     *
     * @return shape string like "[64, 3, 7, 7]"
     */
    public String getShapeString() {
        if (shape == null) {
            return "[]";
        }
        return Arrays.toString(shape);
    }

    /**
     * Check if this is a weight tensor (not a buffer).
     *
     * @return true if this is a weight/parameter tensor
     */
    public boolean isWeight() {
        return !isBuffer && (name.endsWith(".weight") || name.endsWith("_weight"));
    }

    /**
     * Check if this is a bias tensor.
     *
     * @return true if this is a bias tensor
     */
    public boolean isBias() {
        return name.endsWith(".bias") || name.endsWith("_bias");
    }

    /**
     * Check if this is a batch normalization running mean/var buffer.
     *
     * @return true if this is a batch norm buffer
     */
    public boolean isBatchNormBuffer() {
        return name.contains("running_mean") || name.contains("running_var") ||
               name.contains("num_batches_tracked");
    }
}
