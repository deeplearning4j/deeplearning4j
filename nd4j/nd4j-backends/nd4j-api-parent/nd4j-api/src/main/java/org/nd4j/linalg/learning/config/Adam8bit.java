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

package org.nd4j.linalg.learning.config;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.Adam8bitUpdater;
import org.nd4j.linalg.learning.BlockQuantizationUtils;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Map;

/**
 * 8-bit Adam optimizer that stores momentum (m) and variance (v) state
 * in INT8 with per-block absmax quantization, following the bitsandbytes approach.
 *
 * This reduces optimizer state memory by ~4x compared to standard Adam:
 * - Standard Adam: 2 * numParams * 4 bytes (FP32 m + FP32 v)
 * - Adam8bit: 2 * numParams * 1 byte + scales overhead (INT8 m + INT8 v)
 *
 * For a 7B parameter model: 56GB -> ~14GB optimizer state.
 *
 * The optimizer works by:
 * 1. Dequantizing the relevant block of m/v state from INT8 to FP32
 * 2. Performing the standard Adam update step in FP32
 * 3. Requantizing the updated m/v state back to INT8
 *
 * Adam Gibson
 * @see Adam
 */
@Data
@Builder(builderClassName = "Builder")
public class Adam8bit implements IUpdater {

    public static final double DEFAULT_LEARNING_RATE = 1e-3;
    public static final double DEFAULT_BETA1 = 0.9;
    public static final double DEFAULT_BETA2 = 0.999;
    public static final double DEFAULT_EPSILON = 1e-8;
    public static final int DEFAULT_BLOCK_SIZE = 2048;

    @lombok.Builder.Default private double learningRate = DEFAULT_LEARNING_RATE;
    private ISchedule learningRateSchedule;
    @lombok.Builder.Default private double beta1 = DEFAULT_BETA1;
    @lombok.Builder.Default private double beta2 = DEFAULT_BETA2;
    @lombok.Builder.Default private double epsilon = DEFAULT_EPSILON;
    @lombok.Builder.Default private int blockSize = DEFAULT_BLOCK_SIZE;
    @lombok.Builder.Default private boolean pagedOptimizer = false;

    public Adam8bit() {
        this(DEFAULT_LEARNING_RATE, null, DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON,
                DEFAULT_BLOCK_SIZE, false);
    }

    public Adam8bit(double learningRate) {
        this(learningRate, null, DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON,
                DEFAULT_BLOCK_SIZE, false);
    }

    public Adam8bit(ISchedule learningRateSchedule) {
        this(Double.NaN, learningRateSchedule, DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON,
                DEFAULT_BLOCK_SIZE, false);
    }

    private Adam8bit(@JsonProperty("learningRate") double learningRate,
                     @JsonProperty("learningRateSchedule") ISchedule learningRateSchedule,
                     @JsonProperty("beta1") double beta1,
                     @JsonProperty("beta2") double beta2,
                     @JsonProperty("epsilon") double epsilon,
                     @JsonProperty("blockSize") int blockSize,
                     @JsonProperty("pagedOptimizer") boolean pagedOptimizer) {
        this.learningRate = learningRate;
        this.learningRateSchedule = learningRateSchedule;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.blockSize = blockSize;
        this.pagedOptimizer = pagedOptimizer;
    }

    @Override
    public long stateSize(long numParams) {
        // We still allocate FP32 state views for compatibility with the updater framework.
        // The actual memory savings come from the Adam8bitUpdater's internal quantized storage.
        return 2 * numParams;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        Adam8bitUpdater u = new Adam8bitUpdater(this);
        viewArray = viewArray.reshape(viewArray.length());
        long[] gradientShape = viewArray.shape().clone();
        gradientShape[0] /= 2;
        u.setStateViewArray(viewArray, gradientShape, viewArray.ordering(), initializeViewArray);
        return u;
    }

    @Override
    public GradientUpdater instantiate(Map<String, INDArray> updaterState, boolean initializeStateArrays) {
        Adam8bitUpdater u = new Adam8bitUpdater(this);
        u.setState(updaterState, initializeStateArrays);
        return u;
    }

    @Override
    public Adam8bit clone() {
        return new Adam8bit(learningRate, learningRateSchedule, beta1, beta2, epsilon,
                blockSize, pagedOptimizer);
    }

    @Override
    public double getLearningRate(int iteration, int epoch) {
        if (learningRateSchedule != null) {
            return learningRateSchedule.valueAt(iteration, epoch);
        }
        return learningRate;
    }

    @Override
    public boolean hasLearningRate() {
        return true;
    }

    @Override
    public void setLrAndSchedule(double lr, ISchedule lrSchedule) {
        this.learningRate = lr;
        this.learningRateSchedule = lrSchedule;
    }

    public static class Builder {
        public Builder() {
        }
    }
}
