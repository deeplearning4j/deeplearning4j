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
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.RmsPropUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Map;

/**
 * The RMSProp (Root Mean Square Propagation) parameter updater.
 * <p>
 * RMSProp maintains a per-parameter exponential moving average of squared gradients and
 * divides the gradient by the root of that average. This normalizes the gradient magnitude
 * and helps alleviate the diminishing learning rate problem of {@link AdaGrad}.
 * <p>
 * The update rule is:
 * <pre>
 *   E[g^2]_t = rmsDecay * E[g^2]_{t-1} + (1 - rmsDecay) * g_t^2
 *   theta_t  = theta_{t-1} - (alpha / sqrt(E[g^2]_t + epsilon)) * g_t
 * </pre>
 * RMSProp was proposed by Geoffrey Hinton and works well for recurrent networks and
 * non-stationary objectives. It is a practical choice when Adam's bias correction
 * is not needed.
 * <p>
 * Default hyper-parameters:
 * <ul>
 *   <li>Learning rate: {@value #DEFAULT_RMSPROP_LEARNING_RATE}</li>
 *   <li>RMS decay: {@value #DEFAULT_RMSPROP_RMSDECAY}</li>
 *   <li>Epsilon (numerical stability): {@value #DEFAULT_RMSPROP_EPSILON}</li>
 * </ul>
 */
@Data
@Builder(builderClassName = "Builder")
public class RmsProp implements IUpdater {
    /** Default learning rate: {@value}. */
    public static final double DEFAULT_RMSPROP_LEARNING_RATE = 1e-1;
    /** Default epsilon (numerical stability term): {@value}. */
    public static final double DEFAULT_RMSPROP_EPSILON = 1e-8;
    /** Default RMS decay coefficient for the moving average of squared gradients: {@value}. */
    public static final double DEFAULT_RMSPROP_RMSDECAY = 0.95;

    /**
     * Fixed learning rate. Ignored when {@link #learningRateSchedule} is non-null.
     * Default: {@value #DEFAULT_RMSPROP_LEARNING_RATE}.
     */
    @lombok.Builder.Default private double learningRate = DEFAULT_RMSPROP_LEARNING_RATE;

    /**
     * Optional learning rate schedule. When set, the schedule determines the learning rate
     * at each iteration/epoch and the fixed {@link #learningRate} value is not used.
     * Default: {@code null} (use fixed learning rate).
     */
    private ISchedule learningRateSchedule;

    /**
     * Exponential decay rate for the moving average of squared gradients.
     * Controls how quickly old gradient information is forgotten.
     * Must be in (0, 1). Default: {@value #DEFAULT_RMSPROP_RMSDECAY}.
     */
    @lombok.Builder.Default private double rmsDecay = DEFAULT_RMSPROP_RMSDECAY;

    /**
     * Small constant added to the denominator for numerical stability.
     * Default: {@value #DEFAULT_RMSPROP_EPSILON}.
     */
    @lombok.Builder.Default private double epsilon = DEFAULT_RMSPROP_EPSILON;

    public RmsProp(){
        this(DEFAULT_RMSPROP_LEARNING_RATE, null, DEFAULT_RMSPROP_RMSDECAY, DEFAULT_RMSPROP_EPSILON);
    }

    public RmsProp(double learningRate){
        this(learningRate, null, DEFAULT_RMSPROP_RMSDECAY, DEFAULT_RMSPROP_EPSILON);
    }

    public RmsProp(ISchedule learningRateSchedule){
        this(Double.NaN, learningRateSchedule, DEFAULT_RMSPROP_RMSDECAY, DEFAULT_RMSPROP_EPSILON);
    }

    public RmsProp(double learningRate, double rmsDecay, double epsilon){
        this(learningRate, null, rmsDecay, epsilon);
    }

    private RmsProp(@JsonProperty("learningRate") double learningRate,
                    @JsonProperty("learningRateSchedule") ISchedule learningRateSchedule,
                    @JsonProperty("rmsDecay") double rmsDecay,
                    @JsonProperty("epsilon") double epsilon){
        this.learningRate = learningRate;
        this.learningRateSchedule = learningRateSchedule;
        this.rmsDecay = rmsDecay;
        this.epsilon = epsilon;
    }

    @Override
    public long stateSize(long numParams) {
        return numParams;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        RmsPropUpdater u = new RmsPropUpdater(this);
        viewArray = viewArray.reshape(viewArray.length());
        u.setStateViewArray(viewArray, viewArray.shape(), viewArray.ordering(), initializeViewArray);
        return u;
    }

    @Override
    public GradientUpdater instantiate(Map<String, INDArray> updaterState, boolean initializeStateArrays) {
        RmsPropUpdater u = new RmsPropUpdater(this);
        u.setState(updaterState, initializeStateArrays);
        return u;
    }

    @Override
    public RmsProp clone() {
        return new RmsProp(learningRate, learningRateSchedule, rmsDecay, epsilon);
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
