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

package org.nd4j.torchscript.ir;

import lombok.Builder;
import lombok.Data;
import org.nd4j.torchscript.format.TorchScriptDataType;

import java.util.Arrays;

/**
 * Represents a value (tensor or scalar) in the TorchScript computation graph.
 */
@Data
@Builder
public class TorchScriptValue {

    /**
     * Unique name/identifier for this value
     */
    private String name;

    /**
     * Shape of the tensor (null for scalars)
     */
    private long[] shape;

    /**
     * Data type
     */
    private TorchScriptDataType dtype;

    /**
     * Whether this is a model parameter (trainable weight)
     */
    @Builder.Default
    private boolean isParameter = false;

    /**
     * Whether this is a buffer (non-trainable, e.g., running_mean)
     */
    @Builder.Default
    private boolean isBuffer = false;

    /**
     * Whether this is a graph input
     */
    @Builder.Default
    private boolean isInput = false;

    /**
     * Whether this is a graph output
     */
    @Builder.Default
    private boolean isOutput = false;

    /**
     * Whether this requires gradient
     */
    @Builder.Default
    private boolean requiresGrad = false;

    /**
     * The node that produces this value (null for inputs/parameters)
     */
    private String producerNode;

    /**
     * Output index if producer has multiple outputs
     */
    @Builder.Default
    private int producerOutputIndex = 0;

    /**
     * Check if this is a tensor value.
     *
     * @return true if tensor (has shape)
     */
    public boolean isTensor() {
        return shape != null && shape.length > 0;
    }

    /**
     * Check if this is a scalar value.
     *
     * @return true if scalar (no shape or empty shape)
     */
    public boolean isScalar() {
        return shape == null || shape.length == 0;
    }

    /**
     * Get the number of dimensions.
     *
     * @return number of dimensions, 0 for scalars
     */
    public int getRank() {
        return shape != null ? shape.length : 0;
    }

    /**
     * Get the number of elements.
     *
     * @return number of elements, 1 for scalars
     */
    public long getNumElements() {
        if (shape == null || shape.length == 0) {
            return 1;
        }
        long elements = 1;
        for (long dim : shape) {
            elements *= dim;
        }
        return elements;
    }

    /**
     * Get a human-readable shape string.
     *
     * @return shape string
     */
    public String getShapeString() {
        if (shape == null) {
            return "scalar";
        }
        return Arrays.toString(shape);
    }

    /**
     * Create a copy with a new name.
     *
     * @param newName the new name
     * @return copied value with new name
     */
    public TorchScriptValue withName(String newName) {
        return TorchScriptValue.builder()
                .name(newName)
                .shape(shape != null ? shape.clone() : null)
                .dtype(dtype)
                .isParameter(isParameter)
                .isBuffer(isBuffer)
                .isInput(isInput)
                .isOutput(isOutput)
                .requiresGrad(requiresGrad)
                .producerNode(producerNode)
                .producerOutputIndex(producerOutputIndex)
                .build();
    }
}
