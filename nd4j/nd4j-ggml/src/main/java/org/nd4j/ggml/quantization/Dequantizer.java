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
import org.nd4j.linalg.factory.Nd4j;

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
     * Dequantize raw bytes to float array.
     * Limited to Integer.MAX_VALUE elements due to Java array size limits.
     * For larger tensors, use {@link DequantizerFactory#dequantizeToArray} which
     * delegates to the native ggml_dequantize C++ op.
     *
     * @param quantizedData the quantized data bytes
     * @param numElements   the expected number of output elements
     * @return dequantized float array
     */
    float[] dequantize(byte[] quantizedData, long numElements);

    /**
     * Dequantize to an INDArray with the specified shape and target type.
     * Default implementation uses the float[] dequantize method for small tensors.
     * For tensors exceeding Integer.MAX_VALUE elements, use the native
     * ggml_dequantize op via {@link DequantizerFactory#dequantizeToArray}.
     *
     * @param quantizedData the quantized data bytes
     * @param shape         the target shape
     * @param targetType    the target data type
     * @return dequantized INDArray
     */
    default INDArray dequantizeToArray(byte[] quantizedData, long[] shape, DataType targetType) {
        long numElements = 1;
        for (long dim : shape) numElements *= dim;

        if (numElements > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                "Tensor has " + numElements + " elements, exceeding Java array limit. " +
                "Use DequantizerFactory.dequantizeToArray() which delegates to the native " +
                "ggml_dequantize op for large tensors.");
        }

        float[] floatData = dequantize(quantizedData, numElements);
        INDArray array = Nd4j.create(floatData, shape);
        if (targetType != DataType.FLOAT) {
            INDArray casted = array.castTo(targetType);
            array.close();
            array = casted;
        }
        return array;
    }

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
