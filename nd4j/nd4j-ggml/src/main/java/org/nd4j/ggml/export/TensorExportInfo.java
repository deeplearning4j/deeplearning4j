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
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

/**
 * Holds information about a tensor to be exported to GGML format.
 */
@Data
@Builder
public class TensorExportInfo {

    /**
     * Original SameDiff variable name
     */
    private String sdVarName;

    /**
     * Target GGML tensor name (e.g., "blk.0.attn_q.weight")
     */
    private String ggmlName;

    /**
     * The tensor data
     */
    private INDArray array;

    /**
     * Shape of the tensor
     */
    private long[] shape;

    /**
     * Target data type for GGML (may be quantized)
     */
    private GGMLDataType targetDataType;

    /**
     * Byte offset in the data section (computed during writing)
     */
    @Builder.Default
    private long dataOffset = -1;

    /**
     * Size in bytes (computed during writing)
     */
    @Builder.Default
    private long dataSize = -1;

    /**
     * Get the number of dimensions
     */
    public int getNumDimensions() {
        return shape != null ? shape.length : 0;
    }

    /**
     * Get the total number of elements
     */
    public long getNumElements() {
        if (shape == null || shape.length == 0) {
            return 0;
        }
        long total = 1;
        for (long dim : shape) {
            total *= dim;
        }
        return total;
    }

    /**
     * Calculate the expected size in bytes based on target data type
     */
    public long calculateDataSize() {
        if (dataSize > 0) {
            return dataSize;
        }
        return targetDataType.calculateStorageBytes(getNumElements());
    }

    /**
     * Check if this tensor will be quantized
     */
    public boolean isQuantized() {
        return targetDataType.isQuantized();
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
        return String.format("TensorExportInfo{sd='%s', ggml='%s', shape=%s, type=%s}",
                sdVarName, ggmlName, getShapeString(), targetDataType);
    }
}
