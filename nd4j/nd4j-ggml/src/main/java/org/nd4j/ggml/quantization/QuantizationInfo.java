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

import lombok.Builder;
import lombok.Data;
import org.nd4j.ggml.format.GGMLDataType;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Container for quantization metadata extracted from quantized tensors.
 * Used when preserving quantization information during conversion.
 */
@Data
@Builder
public class QuantizationInfo {

    /**
     * The quantization type
     */
    private GGMLDataType quantType;

    /**
     * Block size for the quantization scheme
     */
    private int blockSize;

    /**
     * Scale factors for each block
     */
    private float[] scales;

    /**
     * Zero-point values for asymmetric quantization (may be null)
     */
    private float[] zeroPoints;

    /**
     * Minimum values for Q4_1, Q5_1 style quantization (may be null)
     */
    private float[] mins;

    /**
     * Original tensor shape
     */
    private long[] originalShape;

    /**
     * Number of blocks in the quantized tensor
     */
    private int numBlocks;

    /**
     * Total number of elements in the original tensor
     */
    public long getTotalElements() {
        if (originalShape == null || originalShape.length == 0) {
            return 0;
        }
        long total = 1;
        for (long dim : originalShape) {
            total *= dim;
        }
        return total;
    }

    /**
     * Convert to metadata map for storing in SDZ format
     */
    public Map<String, String> toMetadata() {
        Map<String, String> metadata = new HashMap<>();
        metadata.put("ggml.quant_type", quantType.name());
        metadata.put("ggml.block_size", String.valueOf(blockSize));
        metadata.put("ggml.num_blocks", String.valueOf(numBlocks));
        metadata.put("ggml.original_shape", Arrays.toString(originalShape));
        return metadata;
    }

    /**
     * Calculate the expected number of blocks for a given tensor size
     */
    public static int calculateNumBlocks(long numElements, int blockSize) {
        return (int) ((numElements + blockSize - 1) / blockSize);
    }
}
