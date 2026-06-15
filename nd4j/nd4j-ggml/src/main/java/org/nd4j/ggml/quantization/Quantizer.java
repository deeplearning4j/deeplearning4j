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
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Interface for quantizing tensors to GGML quantized formats.
 * This is the reverse operation of {@link Dequantizer}.
 */
public interface Quantizer {

    /**
     * Get the quantization type this quantizer produces
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
     * Quantize float data to raw bytes
     *
     * @param floatData the input float array
     * @return quantized byte array
     */
    byte[] quantize(float[] floatData);

    /**
     * Quantize an INDArray to raw bytes
     *
     * @param array the input array (will be converted to float if needed)
     * @return quantized byte array
     */
    byte[] quantize(INDArray array);

    /**
     * Calculate the output size in bytes for the given number of elements
     */
    default long calculateOutputBytes(long numElements) {
        int numBlocks = (int) ((numElements + getBlockSize() - 1) / getBlockSize());
        return (long) numBlocks * getBytesPerBlock();
    }

    /**
     * Get quantization statistics for the given data.
     * Useful for evaluating quantization quality.
     *
     * @param floatData the input float data
     * @return statistics about the quantization
     */
    default QuantizationStats getQuantizationStats(float[] floatData) {
        // Quantize and dequantize to compute error
        byte[] quantized = quantize(floatData);

        // Get corresponding dequantizer if available
        if (!DequantizerFactory.hasDequantizer(getQuantType())) {
            return QuantizationStats.builder()
                    .minValue(min(floatData))
                    .maxValue(max(floatData))
                    .build();
        }

        Dequantizer dequantizer = DequantizerFactory.getDequantizer(getQuantType());
        float[] dequantized = dequantizer.dequantize(quantized, floatData.length);

        // Compute error statistics
        float minVal = Float.MAX_VALUE;
        float maxVal = Float.MIN_VALUE;
        double sumAbsError = 0;
        float maxAbsError = 0;
        double sumSquaredError = 0;
        double sumSquaredOriginal = 0;

        for (int i = 0; i < floatData.length; i++) {
            float original = floatData[i];
            float reconstructed = dequantized[i];
            float error = Math.abs(original - reconstructed);

            minVal = Math.min(minVal, original);
            maxVal = Math.max(maxVal, original);
            sumAbsError += error;
            maxAbsError = Math.max(maxAbsError, error);
            sumSquaredError += error * error;
            sumSquaredOriginal += original * original;
        }

        float meanAbsError = (float) (sumAbsError / floatData.length);
        float snr = sumSquaredOriginal > 0 ?
                (float) (10 * Math.log10(sumSquaredOriginal / sumSquaredError)) : Float.POSITIVE_INFINITY;

        return QuantizationStats.builder()
                .minValue(minVal)
                .maxValue(maxVal)
                .meanAbsError(meanAbsError)
                .maxAbsError(maxAbsError)
                .snr(snr)
                .build();
    }

    /**
     * Check if this quantizer can handle the given data range effectively.
     * Some quantizers may have issues with very large or very small values.
     */
    default boolean canHandleRange(float minVal, float maxVal) {
        // Most quantizers can handle any range
        return true;
    }

    // Helper methods
    private static float min(float[] data) {
        float min = Float.MAX_VALUE;
        for (float v : data) {
            min = Math.min(min, v);
        }
        return min;
    }

    private static float max(float[] data) {
        float max = Float.MIN_VALUE;
        for (float v : data) {
            max = Math.max(max, v);
        }
        return max;
    }

    /**
     * Statistics about quantization quality
     */
    @Data
    @Builder
    class QuantizationStats {
        /** Minimum value in the original data */
        private float minValue;
        /** Maximum value in the original data */
        private float maxValue;
        /** Mean absolute error between original and dequantized values */
        private float meanAbsError;
        /** Maximum absolute error */
        private float maxAbsError;
        /** Signal-to-noise ratio in dB */
        private float snr;

        @Override
        public String toString() {
            return String.format("QuantizationStats{range=[%.4f, %.4f], MAE=%.6f, maxErr=%.6f, SNR=%.2fdB}",
                    minValue, maxValue, meanAbsError, maxAbsError, snr);
        }
    }
}
