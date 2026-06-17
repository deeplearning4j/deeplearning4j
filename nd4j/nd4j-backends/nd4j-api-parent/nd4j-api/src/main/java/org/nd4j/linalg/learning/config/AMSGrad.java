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
import org.nd4j.linalg.learning.AMSGradUpdater;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.Map;

/**
 * The AMSGrad parameter updater.
 * <p>
 * AMSGrad is a variant of {@link Adam} that addresses a convergence issue in certain
 * non-convex settings by using the maximum of past squared gradient estimates rather
 * than the exponential moving average. This ensures the effective learning rate is
 * non-increasing over time.
 * <p>
 * The update rule is:
 * <pre>
 *   m_t   = beta1 * m_{t-1} + (1 - beta1) * g_t
 *   v_t   = beta2 * v_{t-1} + (1 - beta2) * g_t^2
 *   v_hat = max(v_hat_{t-1}, v_t)
 *   theta_t = theta_{t-1} - alpha * m_t / (sqrt(v_hat) + epsilon)
 * </pre>
 * AMSGrad can be preferable over Adam when training stability matters or when Adam
 * fails to converge on certain tasks.
 * <p>
 * Reference: <a href="https://arxiv.org/abs/1904.09237">Reddi et al., 2018</a>
 * <p>
 * Default hyper-parameters:
 * <ul>
 *   <li>Learning rate: {@value #DEFAULT_AMSGRAD_LEARNING_RATE}</li>
 *   <li>Beta1 (mean decay): {@value #DEFAULT_AMSGRAD_BETA1_MEAN_DECAY}</li>
 *   <li>Beta2 (variance decay): {@value #DEFAULT_AMSGRAD_BETA2_VAR_DECAY}</li>
 *   <li>Epsilon (numerical stability): {@value #DEFAULT_AMSGRAD_EPSILON}</li>
 * </ul>
 */
@Data
@Builder(builderClassName = "Builder")
public class AMSGrad implements IUpdater {

    /** Default learning rate: {@value}. */
    public static final double DEFAULT_AMSGRAD_LEARNING_RATE = 1e-3;
    /** Default epsilon (numerical stability term): {@value}. */
    public static final double DEFAULT_AMSGRAD_EPSILON = 1e-8;
    /** Default beta1 (exponential decay rate for the first-moment estimate): {@value}. */
    public static final double DEFAULT_AMSGRAD_BETA1_MEAN_DECAY = 0.9;
    /** Default beta2 (exponential decay rate for the second-moment estimate): {@value}. */
    public static final double DEFAULT_AMSGRAD_BETA2_VAR_DECAY = 0.999;

    /**
     * Fixed learning rate. Ignored when {@link #learningRateSchedule} is non-null.
     * Default: {@value #DEFAULT_AMSGRAD_LEARNING_RATE}.
     */
    @lombok.Builder.Default private double learningRate = DEFAULT_AMSGRAD_LEARNING_RATE;

    /**
     * Optional learning rate schedule. When set, the schedule determines the learning rate
     * at each iteration/epoch and the fixed {@link #learningRate} value is not used.
     * Default: {@code null} (use fixed learning rate).
     */
    private ISchedule learningRateSchedule;

    /**
     * Exponential decay rate for the first-moment (mean) estimate of the gradient.
     * Must be in [0, 1). Default: {@value #DEFAULT_AMSGRAD_BETA1_MEAN_DECAY}.
     */
    @lombok.Builder.Default private double beta1 = DEFAULT_AMSGRAD_BETA1_MEAN_DECAY;

    /**
     * Exponential decay rate for the second-moment (uncentered variance) estimate of the gradient.
     * Must be in [0, 1). Default: {@value #DEFAULT_AMSGRAD_BETA2_VAR_DECAY}.
     */
    @lombok.Builder.Default private double beta2 = DEFAULT_AMSGRAD_BETA2_VAR_DECAY;

    /**
     * Small constant added to the denominator for numerical stability.
     * Default: {@value #DEFAULT_AMSGRAD_EPSILON}.
     */
    @lombok.Builder.Default private double epsilon = DEFAULT_AMSGRAD_EPSILON;

    public AMSGrad() {
        this(DEFAULT_AMSGRAD_LEARNING_RATE, DEFAULT_AMSGRAD_BETA1_MEAN_DECAY, DEFAULT_AMSGRAD_BETA2_VAR_DECAY,
                        DEFAULT_AMSGRAD_EPSILON);
    }

    public AMSGrad(double learningRate) {
        this(learningRate, null, DEFAULT_AMSGRAD_BETA1_MEAN_DECAY, DEFAULT_AMSGRAD_BETA2_VAR_DECAY, DEFAULT_AMSGRAD_EPSILON);
    }

    public AMSGrad(ISchedule learningRateSchedule){
        this(Double.NaN, learningRateSchedule, DEFAULT_AMSGRAD_BETA1_MEAN_DECAY, DEFAULT_AMSGRAD_BETA2_VAR_DECAY, DEFAULT_AMSGRAD_EPSILON);
    }

    public AMSGrad(double learningRate, double beta1, double beta2, double epsilon) {
        this(learningRate, null, beta1, beta2, epsilon);
    }

    private AMSGrad(@JsonProperty("learningRate") double learningRate,
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
        return 3 * numParams;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        AMSGradUpdater u = new AMSGradUpdater(this);
        viewArray = viewArray.reshape(viewArray.length());
        long[] gradientShape = viewArray.shape();
        gradientShape = Arrays.copyOf(gradientShape, gradientShape.length);
        gradientShape[0] /= 3;
        u.setStateViewArray(viewArray, gradientShape, viewArray.ordering(), initializeViewArray);
        return u;
    }

    @Override
    public GradientUpdater instantiate(Map<String, INDArray> updaterState, boolean initializeStateArrays) {
        AMSGradUpdater u = new AMSGradUpdater(this);
        u.setState(updaterState, initializeStateArrays);
        return u;
    }

    @Override
    public AMSGrad clone() {
        return new AMSGrad(learningRate, learningRateSchedule, beta1, beta2, epsilon);
    }

    @Override
    public double getLearningRate(int iteration, int epoch) {
        if(learningRateSchedule != null) {
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
