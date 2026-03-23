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

package org.nd4j.linalg.learning;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.Pair;

/**
 * Utility class for block-wise quantization of optimizer state.
 * Quantizes FP32 tensors into INT8 with per-block absmax scaling,
 * following the bitsandbytes approach for 8-bit optimizers.
 *
 * Each block of {@code blockSize} elements is independently quantized
 * using a single absmax scale factor, mapping [-absmax, absmax] to [-127, 127].
 *
 * Adam Gibson
 */
public class BlockQuantizationUtils {

    public static final int DEFAULT_BLOCK_SIZE = 2048;

    private BlockQuantizationUtils() {
    }

    /**
     * Quantize a float tensor into INT8 blocks with per-block scale factors.
     *
     * @param input     The FP32 input tensor (will be flattened)
     * @param blockSize Number of elements per quantization block
     * @return Pair of (quantized INT8 tensor, scale factors FLOAT tensor of shape [numBlocks])
     */
    public static Pair<INDArray, INDArray> quantizeBlocks(INDArray input, int blockSize) {
        Preconditions.checkNotNull(input, "Input must not be null");
        Preconditions.checkArgument(blockSize > 0, "Block size must be positive, got %s", blockSize);

        INDArray flat = input.reshape(input.length());
        long length = flat.length();
        long numBlocks = (length + blockSize - 1) / blockSize;

        INDArray quantized = Nd4j.create(DataType.INT8, length);
        INDArray scales = Nd4j.create(DataType.FLOAT, numBlocks);

        for (long b = 0; b < numBlocks; b++) {
            long start = b * blockSize;
            long end = Math.min(start + blockSize, length);

            // Find absmax in this block
            double absmax = 0.0;
            for (long i = start; i < end; i++) {
                double val = Math.abs(flat.getDouble(i));
                if (val > absmax) {
                    absmax = val;
                }
            }

            double scale = absmax > 0 ? absmax / 127.0 : 1.0;
            scales.putScalar(b, scale);

            // Quantize: round(value / scale) clamped to [-127, 127]
            for (long i = start; i < end; i++) {
                int q = (int) Math.round(flat.getDouble(i) / scale);
                q = Math.max(-127, Math.min(127, q));
                quantized.putScalar(i, q);
            }
        }

        return Pair.of(quantized, scales);
    }

    /**
     * Dequantize INT8 blocks back to FP32 using per-block scale factors.
     *
     * @param quantized The INT8 quantized tensor
     * @param scales    Per-block scale factors (FLOAT, shape [numBlocks])
     * @param blockSize Number of elements per block
     * @return Dequantized FP32 tensor with the same shape as quantized
     */
    public static INDArray dequantizeBlocks(INDArray quantized, INDArray scales, int blockSize) {
        Preconditions.checkNotNull(quantized, "Quantized tensor must not be null");
        Preconditions.checkNotNull(scales, "Scales must not be null");

        INDArray flat = quantized.reshape(quantized.length());
        long length = flat.length();
        INDArray output = Nd4j.create(DataType.FLOAT, length);

        long numBlocks = scales.length();
        for (long b = 0; b < numBlocks; b++) {
            long start = b * blockSize;
            long end = Math.min(start + blockSize, length);
            double scale = scales.getDouble(b);

            for (long i = start; i < end; i++) {
                output.putScalar(i, flat.getDouble(i) * scale);
            }
        }

        return output;
    }

    /**
     * Quantize using default block size.
     */
    public static Pair<INDArray, INDArray> quantizeBlocks(INDArray input) {
        return quantizeBlocks(input, DEFAULT_BLOCK_SIZE);
    }

    /**
     * Calculate the number of blocks for a given tensor length and block size.
     */
    public static long numBlocks(long length, int blockSize) {
        return (length + blockSize - 1) / blockSize;
    }

    /**
     * Calculate the total state memory for 8-bit optimizer state.
     * Each parameter needs 1 byte (INT8) + scale overhead.
     *
     * @param numParams Total number of parameters
     * @param blockSize Block size for quantization
     * @param numStates Number of optimizer states (e.g., 2 for Adam: m and v)
     * @return Total bytes needed
     */
    public static long stateMemoryBytes(long numParams, int blockSize, int numStates) {
        long blocks = numBlocks(numParams, blockSize);
        // INT8 state + FLOAT32 scales per block
        long perState = numParams + blocks * 4; // 1 byte per param + 4 bytes per scale
        return perState * numStates;
    }
}
