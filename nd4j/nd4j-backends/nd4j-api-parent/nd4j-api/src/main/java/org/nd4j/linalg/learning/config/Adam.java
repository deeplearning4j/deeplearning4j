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
import org.nd4j.linalg.learning.AdamUpdater;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.Map;

/**
 * The Adam (Adaptive Moment Estimation) parameter updater.
 * <p>
 * Adam maintains per-parameter first-moment (mean) and second-moment (uncentered variance)
 * estimates of the gradients and uses them to scale the effective learning rate for each
 * parameter individually. The update rule is:
 * <pre>
 *   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
 *   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
 *   m_hat = m_t / (1 - beta1^t)
 *   v_hat = v_t / (1 - beta2^t)
 *   theta_t = theta_{t-1} - alpha * m_hat / (sqrt(v_hat) + epsilon)
 * </pre>
 * Adam is a good general-purpose optimizer and is suitable for most deep-learning problems.
 * <p>
 * Reference: <a href="https://arxiv.org/abs/1412.6980">Kingma and Ba, 2014</a>
 * <p>
 * Default hyper-parameters:
 * <ul>
 *   <li>Learning rate ({@code alpha}): {@value #DEFAULT_ADAM_LEARNING_RATE}</li>
 *   <li>Beta1 (mean decay): {@value #DEFAULT_ADAM_BETA1_MEAN_DECAY}</li>
 *   <li>Beta2 (variance decay): {@value #DEFAULT_ADAM_BETA2_VAR_DECAY}</li>
 *   <li>Epsilon (numerical stability): {@value #DEFAULT_ADAM_EPSILON}</li>
 * </ul>
 *
 * @author Adam Gibson
 */
@Data
@Builder(builderClassName = "Builder")
public class Adam implements IUpdater {

    /** Default learning rate: {@value}. */
    public static final double DEFAULT_ADAM_LEARNING_RATE = 1e-3;
    /** Default epsilon (numerical stability term added to the denominator): {@value}. */
    public static final double DEFAULT_ADAM_EPSILON = 1e-8;
    /** Default beta1 (exponential decay rate for the first-moment estimate): {@value}. */
    public static final double DEFAULT_ADAM_BETA1_MEAN_DECAY = 0.9;
    /** Default beta2 (exponential decay rate for the second-moment estimate): {@value}. */
    public static final double DEFAULT_ADAM_BETA2_VAR_DECAY = 0.999;

    /**
     * Fixed learning rate. Ignored when {@link #learningRateSchedule} is non-null.
     * Default: {@value #DEFAULT_ADAM_LEARNING_RATE}.
     */
    @lombok.Builder.Default private double learningRate = DEFAULT_ADAM_LEARNING_RATE;

    /**
     * Optional learning rate schedule. When set, the schedule determines the learning rate
     * at each iteration/epoch and the fixed {@link #learningRate} value is not used.
     * Default: {@code null} (use fixed learning rate).
     */
    private ISchedule learningRateSchedule;

    /**
     * Exponential decay rate for the first-moment (mean) estimate of the gradient.
     * Must be in [0, 1). Default: {@value #DEFAULT_ADAM_BETA1_MEAN_DECAY}.
     */
    @lombok.Builder.Default private double beta1 = DEFAULT_ADAM_BETA1_MEAN_DECAY;

    /**
     * Exponential decay rate for the second-moment (uncentered variance) estimate of the gradient.
     * Must be in [0, 1). Default: {@value #DEFAULT_ADAM_BETA2_VAR_DECAY}.
     */
    @lombok.Builder.Default private double beta2 = DEFAULT_ADAM_BETA2_VAR_DECAY;

    /**
     * Small constant added to the denominator for numerical stability.
     * Default: {@value #DEFAULT_ADAM_EPSILON}.
     */
    @lombok.Builder.Default private double epsilon = DEFAULT_ADAM_EPSILON;

    public Adam() {
        this(DEFAULT_ADAM_LEARNING_RATE, DEFAULT_ADAM_BETA1_MEAN_DECAY, DEFAULT_ADAM_BETA2_VAR_DECAY,
                        DEFAULT_ADAM_EPSILON);
    }

    public Adam(double learningRate){
        this(learningRate, null, DEFAULT_ADAM_BETA1_MEAN_DECAY, DEFAULT_ADAM_BETA2_VAR_DECAY, DEFAULT_ADAM_EPSILON);
    }

    public Adam(ISchedule learningRateSchedule){
        this(Double.NaN, learningRateSchedule, DEFAULT_ADAM_BETA1_MEAN_DECAY, DEFAULT_ADAM_BETA2_VAR_DECAY, DEFAULT_ADAM_EPSILON);
    }

    public Adam(double learningRate, double beta1, double beta2, double epsilon) {
        this(learningRate, null, beta1, beta2, epsilon);
    }

    private Adam(@JsonProperty("learningRate") double learningRate,
                 @JsonProperty("learningRateSchedule") ISchedule learningRateSchedule,
                 @JsonProperty("beta1") double beta1,
                 @JsonProperty("beta2") double beta2,
                 @JsonProperty("epsilon") double epsilon){
        this.learningRate = learningRate;
        this.learningRateSchedule = learningRateSchedule;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    @Override
    public long stateSize(long numParams) {
        return 2 * numParams;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        AdamUpdater u = new AdamUpdater(this);
        viewArray = viewArray.reshape(viewArray.length());
        long[] gradientShape = viewArray.shape();
        gradientShape = Arrays.copyOf(gradientShape, gradientShape.length);
        gradientShape[0] /= 2;
        u.setStateViewArray(viewArray, gradientShape, viewArray.ordering(), initializeViewArray);
        return u;
    }

    @Override
    public GradientUpdater instantiate(Map<String, INDArray> updaterState, boolean initializeStateArrays) {
        AdamUpdater u = new AdamUpdater(this);
        u.setState(updaterState, initializeStateArrays);
        return u;
    }

    @Override
    public Adam clone() {
        return new Adam(learningRate, learningRateSchedule, beta1, beta2, epsilon);
    }

    @Override
    public double getLearningRate(int iteration, int epoch){
        if(learningRateSchedule != null){
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

    //Partial builder implementation to give public no-arg constructor
    public static class Builder {
        public Builder(){ }
    }
}
