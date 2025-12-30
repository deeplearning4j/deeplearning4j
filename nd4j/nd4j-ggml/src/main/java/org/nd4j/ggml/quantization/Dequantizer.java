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

import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Interface for dequantizing GGML quantized tensors to full precision.
 */
public interface Dequantizer {

    /**
     * Get the quantization type this dequantizer handles
     */
    GGMLDataType getQuantType();

    /**
     * Get the block size for this quantization type
     */
    int getBlockSize();

    /**
     * Get the number of bytes per block
     */
    int getBytesPerBlock();

    /**
     * Dequantize raw bytes to float array
     *
     * @param quantizedData the quantized data bytes
     * @param numElements   the expected number of output elements
     * @return dequantized float array
     */
    float[] dequantize(byte[] quantizedData, long numElements);

    /**
     * Dequantize to an INDArray with the specified shape and target type
     *
     * @param quantizedData the quantized data bytes
     * @param shape         the target shape
     * @param targetType    the target data type
     * @return dequantized INDArray
     */
    INDArray dequantizeToArray(byte[] quantizedData, long[] shape, DataType targetType);

    /**
     * Extract quantization metadata from the raw data
     *
     * @param quantizedData the quantized data bytes
     * @param shape         the original tensor shape
     * @return quantization information
     */
    QuantizationInfo extractQuantizationInfo(byte[] quantizedData, long[] shape);

    /**
     * Calculate the number of bytes needed to store elements of this type
     *
     * @param numElements number of elements
     * @return bytes needed
     */
    default long calculateStorageBytes(long numElements) {
        int numBlocks = (int) ((numElements + getBlockSize() - 1) / getBlockSize());
        return (long) numBlocks * getBytesPerBlock();
    }
}
