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

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.NesterovsUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Map;

/**
 * The Nesterov Accelerated Gradient (NAG) SGD updater.
 * <p>
 * Nesterovs is a variant of momentum-based stochastic gradient descent that computes the
 * gradient at a "look-ahead" position rather than the current position. This typically
 * results in faster convergence and better handling of sharp curvature compared to
 * classical momentum SGD.
 * <p>
 * The update rule is:
 * <pre>
 *   v_t = momentum * v_{t-1} - alpha * grad(theta_{t-1} + momentum * v_{t-1})
 *   theta_t = theta_{t-1} + v_t
 * </pre>
 * Nesterovs is a strong baseline for many supervised learning tasks and is often
 * preferable to plain SGD with momentum when the learning rate needs careful tuning.
 * <p>
 * Default hyper-parameters:
 * <ul>
 *   <li>Learning rate: {@value #DEFAULT_NESTEROV_LEARNING_RATE}</li>
 *   <li>Momentum: {@value #DEFAULT_NESTEROV_MOMENTUM}</li>
 * </ul>
 */
@AllArgsConstructor
@Data
@Builder(builderClassName = "Builder")
public class Nesterovs implements IUpdater {
    /** Default momentum coefficient: {@value}. */
    public static final double DEFAULT_NESTEROV_MOMENTUM = 0.9;
    /** Default learning rate: {@value}. */
    public static final double DEFAULT_NESTEROV_LEARNING_RATE = 0.1;

    /**
     * Fixed learning rate. Ignored when {@link #learningRateSchedule} is non-null.
     * Default: {@value #DEFAULT_NESTEROV_LEARNING_RATE}.
     */
    @lombok.Builder.Default private double learningRate = DEFAULT_NESTEROV_LEARNING_RATE;

    /**
     * Optional learning rate schedule. When set, the schedule determines the learning rate
     * at each iteration/epoch and the fixed {@link #learningRate} value is not used.
     * Default: {@code null} (use fixed learning rate).
     */
    private ISchedule learningRateSchedule;

    /**
     * Fixed momentum coefficient. Ignored when {@link #momentumISchedule} is non-null.
     * Controls how much of the previous velocity is retained each step.
     * Default: {@value #DEFAULT_NESTEROV_MOMENTUM}.
     */
    @lombok.Builder.Default private double momentum = DEFAULT_NESTEROV_MOMENTUM;

    /**
     * Optional momentum schedule. When set, the momentum value at each iteration/epoch
     * is determined by the schedule and the fixed {@link #momentum} value is not used.
     * Default: {@code null} (use fixed momentum).
     */
    private ISchedule momentumISchedule;

    /**
     * @deprecated Use {@link #momentumISchedule} with an {@link ISchedule} instead.
     */
    @Deprecated
    private Map<Integer,Double> momentumSchedule;

    public Nesterovs(){
        this(DEFAULT_NESTEROV_LEARNING_RATE, null, DEFAULT_NESTEROV_MOMENTUM, null);
    }

    public Nesterovs(double momentum) {
        this(DEFAULT_NESTEROV_LEARNING_RATE, momentum);
    }

    public Nesterovs(double learningRate, double momentum){
        this(learningRate, null, momentum, null);
    }

    public Nesterovs(ISchedule learningRateSchedule){
        this(Double.NaN, learningRateSchedule, DEFAULT_NESTEROV_MOMENTUM, null);
    }

    public Nesterovs(ISchedule learningRateSchedule, double momentum){
        this(Double.NaN, learningRateSchedule, momentum, null);
    }

    public Nesterovs(ISchedule learningRateSchedule, ISchedule momentumSchedule){
        this(Double.NaN, learningRateSchedule, Double.NaN, momentumSchedule);
    }

    public Nesterovs(double learningRate, ISchedule momentumSchedule){
        this(learningRate, null, Double.NaN, momentumSchedule);
    }

    private Nesterovs(@JsonProperty("learningRate") double learningRate,
                      @JsonProperty("learningRateSchedule") ISchedule learningRateSchedule,
                      @JsonProperty("momentum") double momentum,
                      @JsonProperty("momentumSchedule") ISchedule momentumISchedule){
        this.learningRate = learningRate;
        this.learningRateSchedule = learningRateSchedule;
        this.momentum = momentum;
        this.momentumISchedule = momentumISchedule;
    }

    @Override
    public long stateSize(long numParams) {
        return numParams;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        NesterovsUpdater u = new NesterovsUpdater(this);
        viewArray = viewArray.reshape(viewArray.length());
        u.setStateViewArray(viewArray, viewArray.shape(), viewArray.ordering(), initializeViewArray);
        return u;
    }

    @Override
    public GradientUpdater instantiate(Map<String, INDArray> updaterState, boolean initializeStateArrays) {
        NesterovsUpdater u = new NesterovsUpdater(this);
        u.setState(updaterState, initializeStateArrays);
        return u;
    }

    @Override
    public Nesterovs clone() {
        return new Nesterovs(learningRate, learningRateSchedule, momentum, momentumISchedule);
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

    public double currentMomentum(int iteration, int epoch){
        if(momentumISchedule != null){
            return momentumISchedule.valueAt(iteration, epoch);
        }
        return momentum;
    }

    //Partial builder implementation to give public no-arg constructor
    public static class Builder {
        public Builder(){ }
    }
}
