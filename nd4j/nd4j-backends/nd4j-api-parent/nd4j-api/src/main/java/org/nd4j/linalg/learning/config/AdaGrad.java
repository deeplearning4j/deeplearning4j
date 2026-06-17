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
import org.nd4j.linalg.learning.AdaGradUpdater;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Map;

/**
 * The AdaGrad (Adaptive Gradient) parameter updater.
 * <p>
 * AdaGrad adapts the learning rate for each parameter by accumulating the sum of squared
 * historical gradients. Parameters that receive large or frequent gradient updates will see
 * a progressively smaller effective learning rate, while infrequently updated parameters
 * retain a larger effective learning rate.
 * <p>
 * The update rule is:
 * <pre>
 *   G_t = G_{t-1} + g_t^2
 *   theta_t = theta_{t-1} - (alpha / sqrt(G_t + epsilon)) * g_t
 * </pre>
 * AdaGrad is well suited for sparse data (e.g. NLP tasks with word embeddings) but can
 * suffer from a monotonically decreasing learning rate that becomes too small over time.
 * Consider {@link RmsProp} or {@link Adam} if training stalls.
 * <p>
 * Default hyper-parameters:
 * <ul>
 *   <li>Learning rate ({@code alpha}): {@value #DEFAULT_ADAGRAD_LEARNING_RATE}</li>
 *   <li>Epsilon (numerical stability): {@value #DEFAULT_ADAGRAD_EPSILON}</li>
 * </ul>
 */
@Data
@Builder(builderClassName = "Builder")
public class AdaGrad implements IUpdater {

    /** Default learning rate: {@value}. */
    public static final double DEFAULT_ADAGRAD_LEARNING_RATE = 1e-1;
    /** Default epsilon (numerical stability term added to the denominator): {@value}. */
    public static final double DEFAULT_ADAGRAD_EPSILON = 1e-6;

    /**
     * Fixed learning rate. Ignored when {@link #learningRateSchedule} is non-null.
     * Default: {@value #DEFAULT_ADAGRAD_LEARNING_RATE}.
     */
    @lombok.Builder.Default private double learningRate = DEFAULT_ADAGRAD_LEARNING_RATE;

    /**
     * Optional learning rate schedule. When set, the schedule determines the learning rate
     * at each iteration/epoch and the fixed {@link #learningRate} value is not used.
     * Default: {@code null} (use fixed learning rate).
     */
    private ISchedule learningRateSchedule;

    /**
     * Small constant added to the denominator for numerical stability.
     * Default: {@value #DEFAULT_ADAGRAD_EPSILON}.
     */
    @lombok.Builder.Default private double epsilon = DEFAULT_ADAGRAD_EPSILON;

    public AdaGrad(){
        this(DEFAULT_ADAGRAD_LEARNING_RATE, null, DEFAULT_ADAGRAD_EPSILON);
    }

    public AdaGrad(double learningRate){
        this(learningRate, null, DEFAULT_ADAGRAD_EPSILON);
    }

    public AdaGrad(double learningRate, double epsilon){
        this(learningRate, null, epsilon);
    }

    public AdaGrad(ISchedule learningRateSchedule){
        this(Double.NaN, learningRateSchedule, DEFAULT_ADAGRAD_EPSILON);
    }

    public AdaGrad(ISchedule learningRateSchedule, double epsilon){
        this(Double.NaN, learningRateSchedule, epsilon);
    }

    private AdaGrad(@JsonProperty("learningRate") double learningRate,
                    @JsonProperty("learningRateSchedule") ISchedule learningRateSchedule,
                    @JsonProperty("epsilon") double epsilon){
        this.learningRate = learningRate;
        this.learningRateSchedule = learningRateSchedule;
        this.epsilon = epsilon;
    }

    @Override
    public long stateSize(long numParams) {
        return numParams;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
       viewArray = viewArray.reshape(viewArray.length());
        AdaGradUpdater u = new AdaGradUpdater(this);
        u.setStateViewArray(viewArray, viewArray.shape(), viewArray.ordering(), initializeViewArray);
        return u;
    }

    @Override
    public GradientUpdater instantiate(Map<String, INDArray> updaterState, boolean initializeStateArrays) {
        AdaGradUpdater u = new AdaGradUpdater(this);
        u.setState(updaterState, initializeStateArrays);
        return u;
    }

    @Override
    public AdaGrad clone() {
        return new AdaGrad(learningRate, epsilon);
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
