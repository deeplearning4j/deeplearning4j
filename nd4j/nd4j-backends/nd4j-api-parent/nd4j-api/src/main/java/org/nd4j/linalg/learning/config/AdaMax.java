/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.learning.config;

import lombok.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AdaMaxUpdater;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;

/**
 * The AdaMax updater, a variant of Adam.
 * http://arxiv.org/abs/1412.6980
 *
 * @author Justin Long
 */
@Data
@Builder(builderClassName = "Builder")
public class AdaMax implements IUpdater {
    public static final double DEFAULT_ADAMAX_LEARNING_RATE = 1e-3;
    public static final double DEFAULT_ADAMAX_EPSILON = 1e-8;
    public static final double DEFAULT_ADAMAX_BETA1_MEAN_DECAY = 0.9;
    public static final double DEFAULT_ADAMAX_BETA2_VAR_DECAY = 0.999;

    @lombok.Builder.Default private double learningRate = DEFAULT_ADAMAX_LEARNING_RATE; // learning rate
    private ISchedule learningRateSchedule;
    @lombok.Builder.Default private double beta1 = DEFAULT_ADAMAX_BETA1_MEAN_DECAY; // gradient moving avg decay rate
    @lombok.Builder.Default private double beta2 = DEFAULT_ADAMAX_BETA2_VAR_DECAY; // gradient sqrd decay rate
    @lombok.Builder.Default private double epsilon = DEFAULT_ADAMAX_EPSILON;

    public AdaMax(){
        this(DEFAULT_ADAMAX_LEARNING_RATE);
    }

    public AdaMax(double learningRate){
        this(learningRate, null, DEFAULT_ADAMAX_BETA1_MEAN_DECAY, DEFAULT_ADAMAX_BETA2_VAR_DECAY, DEFAULT_ADAMAX_EPSILON);
    }

    public AdaMax(ISchedule learningRateSchedule){
        this(Double.NaN, learningRateSchedule, DEFAULT_ADAMAX_BETA1_MEAN_DECAY, DEFAULT_ADAMAX_BETA2_VAR_DECAY, DEFAULT_ADAMAX_EPSILON);
    }

    public AdaMax(double learningRate, double beta1, double beta2, double epsilon){
        this(learningRate, null, beta1, beta2, epsilon);
    }


    private AdaMax(@JsonProperty("learningRate") double learningRate,
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
        AdaMaxUpdater a = new AdaMaxUpdater(this);
        long[] gradientShape = viewArray.shape();
        gradientShape = Arrays.copyOf(gradientShape, gradientShape.length);
        gradientShape[1] /= 2;
        a.setStateViewArray(viewArray, gradientShape, viewArray.ordering(), initializeViewArray);
        return a;
    }

    @Override
    public IUpdater clone() {
        return new AdaMax(learningRate, learningRateSchedule, beta1, beta2, epsilon);
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
