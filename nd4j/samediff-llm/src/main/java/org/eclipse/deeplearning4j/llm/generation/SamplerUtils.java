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

package org.eclipse.deeplearning4j.llm.generation;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Random;

/**
 * Utility methods for token sampling operations.
 * Vectorized implementations using ND4J operations.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public final class SamplerUtils {

    private static final double NEG_INF = Double.NEGATIVE_INFINITY;

    private SamplerUtils() {}

    /**
     * Apply temperature scaling to logits.
     */
    public static INDArray applyTemperature(INDArray logits, double temperature) {
        if (temperature <= 0) {
            throw new IllegalArgumentException("Temperature must be positive, got: " + temperature);
        }
        if (temperature == 1.0) {
            return logits;
        }
        return logits.div(temperature);
    }

    /**
     * Convert logits to probabilities using softmax.
     */
    public static INDArray softmax(INDArray logits) {
        if (logits.rank() == 1) {
            double maxVal = logits.maxNumber().doubleValue();
            INDArray shifted = logits.sub(maxVal);
            INDArray expShifted = Transforms.exp(shifted);
            double sumExp = expShifted.sumNumber().doubleValue();
            return expShifted.div(sumExp);
        } else {
            INDArray maxVal = logits.max(true, 1);
            INDArray shifted = logits.sub(maxVal);
            INDArray expShifted = Transforms.exp(shifted);
            INDArray sumExp = expShifted.sum(true, 1);
            return expShifted.div(sumExp);
        }
    }

    /**
     * Apply top-K filtering to logits.
     */
    public static INDArray topKFilter(INDArray logits, int k) {
        if (k <= 0) {
            return logits;
        }

        long vocabSize = logits.rank() == 1 ? logits.length() : logits.size(1);
        if (k >= vocabSize) {
            return logits;
        }

        boolean was1D = logits.rank() == 1;
        INDArray input = was1D ? logits.reshape(1, logits.length()) : logits;
        INDArray result = input.dup();

        long batchSize = result.size(0);
        for (int b = 0; b < batchSize; b++) {
            INDArray row = result.getRow(b);

            // Sort to find k-th largest value
            INDArray sorted = Nd4j.sort(row.dup(), false);
            double threshold = sorted.getDouble(k - 1);

            // Set values below threshold to -inf using BooleanIndexing
            BooleanIndexing.replaceWhere(row, NEG_INF, Conditions.lessThan(threshold));
        }

        return was1D ? result.reshape(result.length()) : result;
    }

    /**
     * Apply top-P (nucleus) filtering to logits.
     */
    public static INDArray topPFilter(INDArray logits, double p) {
        if (p <= 0 || p >= 1.0) {
            return logits;
        }

        boolean was1D = logits.rank() == 1;
        INDArray input = was1D ? logits.reshape(1, logits.length()) : logits;
        INDArray result = input.dup();

        long batchSize = result.size(0);
        for (int b = 0; b < batchSize; b++) {
            INDArray row = result.getRow(b);

            // Get probabilities for this row
            INDArray probs = softmax(row);

            // Sort probabilities descending and get cumsum
            INDArray sortedProbs = Nd4j.sort(probs.dup(), false);
            INDArray cumsum = sortedProbs.cumsum(0);

            // Find threshold: first index where cumsum > p, then get that prob value
            // Transfer cumsum to host once, iterate on host
            float[] cumsumArr = cumsum.toFloatVector();
            int cutoffIdx = cumsumArr.length - 1;
            for (int i = 0; i < cumsumArr.length; i++) {
                if (cumsumArr[i] > p) {
                    cutoffIdx = i;
                    break;
                }
            }

            float threshold = sortedProbs.toFloatVector()[cutoffIdx];

            // Bulk transfer probs and row to host, apply mask, write back
            float[] probArr = probs.toFloatVector();
            float[] rowArr = row.toFloatVector();
            for (int i = 0; i < rowArr.length; i++) {
                if (probArr[i] < threshold) {
                    rowArr[i] = (float) NEG_INF;
                }
            }
            row.assign(Nd4j.createFromArray(rowArr));
        }

        return was1D ? result.reshape(result.length()) : result;
    }

    /**
     * Apply repetition penalty.
     */
    public static INDArray applyRepetitionPenalty(INDArray logits, int[] generatedTokens, double penalty) {
        if (penalty == 1.0 || generatedTokens == null || generatedTokens.length == 0) {
            return logits;
        }

        INDArray result = logits.dup();
        INDArray flat = result.rank() == 1 ? result : result.getRow(0);
        long vocabSize = flat.length();

        for (int tokenId : generatedTokens) {
            if (tokenId >= 0 && tokenId < vocabSize) {
                double logit = flat.getDouble(tokenId);
                if (logit > 0) {
                    flat.putScalar(tokenId, logit / penalty);
                } else {
                    flat.putScalar(tokenId, logit * penalty);
                }
            }
        }

        return result;
    }

    /**
     * Sample a token using cumulative probability.
     * Transfers data to host once rather than per-element GPU reads.
     */
    public static int multinomialSample(INDArray probs, Random random) {
        INDArray flat = probs.rank() == 1 ? probs : probs.reshape(probs.length());

        // Transfer all probabilities to host at once
        float[] probArr = flat.toFloatVector();
        double r = random.nextDouble();
        double cumsum = 0.0;

        for (int i = 0; i < probArr.length; i++) {
            cumsum += probArr[i];
            if (r <= cumsum) {
                return i;
            }
        }

        return probArr.length - 1;
    }

    /**
     * Batch multinomial sampling.
     */
    public static int[] multinomialSampleBatch(INDArray probs, Random random) {
        if (probs.rank() == 1) {
            return new int[]{multinomialSample(probs, random)};
        }

        int batchSize = (int) probs.size(0);
        int[] result = new int[batchSize];

        for (int b = 0; b < batchSize; b++) {
            result[b] = multinomialSample(probs.getRow(b), random);
        }

        return result;
    }

    /**
     * Argmax - find the index of the maximum value.
     *
     * Uses bulk host transfer (toFloatVector) instead of per-element getFloat()
     * to avoid O(vocabSize) JNI round-trips. The commit() ensures CUDA kernels
     * complete before the D2H transfer, and toFloatVector() does a single bulk
     * copy of the entire array to host memory.
     */
    public static int argmax(INDArray logits) {
        INDArray flat = logits.rank() == 1 ? logits : logits.reshape(logits.length());
        Nd4j.getExecutioner().commit();
        float[] data = flat.toFloatVector();
        int maxIdx = 0;
        float maxVal = data[0];
        for (int i = 1; i < data.length; i++) {
            if (data[i] > maxVal) {
                maxVal = data[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    /**
     * Batch argmax - find the index of the maximum value for each row.
     *
     * Uses GPU-side argmax (Nd4j.argMax) to avoid transferring the entire logits
     * tensor to host. Only the result indices (one int per batch) are transferred.
     * Input is dup'd to ensure contiguity (argMax has issues with views).
     */
    public static int[] argmaxBatch(INDArray logits) {
        if (logits.rank() == 1) {
            return new int[]{argmax(logits)};
        }
        // Ensure contiguous — argMax has issues with views/non-contiguous arrays
        INDArray contiguous = logits.isView() ? logits.dup() : logits;
        try {
            INDArray indices = Nd4j.argMax(contiguous, 1);
            Nd4j.getExecutioner().commit();
            int batchSize = (int) indices.length();
            int[] result = new int[batchSize];
            for (int b = 0; b < batchSize; b++) {
                result[b] = indices.getInt(b);
            }
            indices.close();
            return result;
        } finally {
            if (contiguous != logits) {
                contiguous.close();
            }
        }
    }

    /**
     * Normalize logits.
     */
    public static INDArray normalizeLogits(INDArray logits) {
        if (logits.rank() == 1) {
            return logits.sub(logits.maxNumber());
        } else {
            INDArray maxVal = logits.max(true, 1);
            return logits.sub(maxVal);
        }
    }

    /**
     * Log-softmax computation.
     */
    public static INDArray logSoftmax(INDArray logits) {
        boolean was1D = logits.rank() == 1;
        INDArray input = was1D ? logits.reshape(1, logits.length()) : logits;

        INDArray maxVal = input.max(true, 1);
        INDArray shifted = input.sub(maxVal);
        INDArray expShifted = Transforms.exp(shifted);
        INDArray sumExp = expShifted.sum(true, 1);
        INDArray logSumExp = Transforms.log(sumExp);
        INDArray result = shifted.sub(logSumExp);

        return was1D ? result.reshape(result.length()) : result;
    }

    /**
     * Entropy computation: H = -sum(p * log(p))
     */
    public static double entropy(INDArray probs) {
        INDArray flat = probs.rank() == 1 ? probs : probs.reshape(probs.length());
        INDArray logProbs = Transforms.log(flat.add(1e-10));
        INDArray pLogP = flat.mul(logProbs);
        return -pLogP.sumNumber().doubleValue();
    }

    /**
     * Check if probability distribution is valid.
     */
    public static boolean isValidDistribution(INDArray probs, double tolerance) {
        double sum = probs.sumNumber().doubleValue();
        double min = probs.minNumber().doubleValue();
        return min >= 0 && Math.abs(sum - 1.0) <= tolerance;
    }

    /**
     * Get top-K indices and values.
     */
    public static INDArray[] topK(INDArray logits, int k) {
        k = (int) Math.min(k, logits.length());

        INDArray[] sortResult = Nd4j.sortWithIndices(logits.dup(), 0, false);
        INDArray sortedIndices = sortResult[0];
        INDArray sortedValues = sortResult[1];

        INDArray topIndices = sortedIndices.get(NDArrayIndex.interval(0, k));
        INDArray topValues = sortedValues.get(NDArrayIndex.interval(0, k));

        return new INDArray[]{topIndices, topValues};
    }
}
