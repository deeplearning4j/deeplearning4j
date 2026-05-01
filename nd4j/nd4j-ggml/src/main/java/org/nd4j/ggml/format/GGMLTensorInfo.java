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

import lombok.Builder;
import lombok.Data;

import java.util.Arrays;

/**
 * Information about a tensor stored in a GGML/GGUF file.
 */
@Data
@Builder
public class GGMLTensorInfo {

    /**
     * Name of the tensor (e.g., "blk.0.attn_q.weight")
     */
    private String name;

    /**
     * Number of dimensions
     */
    private int numDimensions;

    /**
     * Shape of the tensor (dimensions)
     */
    private long[] shape;

    /**
     * Data type of the tensor
     */
    private GGMLDataType dataType;

    /**
     * Offset of the tensor data from the start of the data section
     */
    private long dataOffset;

    /**
     * Calculate the total number of elements in this tensor
     */
    public long getNumElements() {
        if (shape == null || shape.length == 0) {
            // Rank-0 scalars have exactly 1 element
            return 1;
        }
        long total = 1;
        for (long dim : shape) {
            total *= dim;
        }
        return total;
    }

    /**
     * Calculate the size in bytes for this tensor's data
     */
    public long getDataSize() {
        return dataType.calculateStorageBytes(getNumElements());
    }

    /**
     * Check if this tensor contains quantized data
     */
    public boolean isQuantized() {
        return dataType.isQuantized();
    }

    /**
     * Get a human-readable shape string
     */
    public String getShapeString() {
        if (shape == null || shape.length == 0) {
            return "[]";
        }
        return Arrays.toString(shape);
    }

    @Override
    public String toString() {
        return String.format("GGMLTensorInfo{name='%s', shape=%s, type=%s, offset=%d}",
                name, getShapeString(), dataType, dataOffset);
    }
}
