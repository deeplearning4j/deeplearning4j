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

package org.deeplearning4j.nn.updater.custom;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.schedule.ISchedule;

/**
 * Created by Alex on 09/05/2017.
 */
@AllArgsConstructor
@Data
public class CustomIUpdater implements IUpdater {

    public static final double DEFAULT_SGD_LR = 1e-3;

    private double learningRate;


    public CustomIUpdater() {
        this(DEFAULT_SGD_LR);
    }

    @Override
    public long stateSize(long numParams) {
        return 0;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        if (viewArray != null) {
            throw new IllegalStateException("View arrays are not supported/required for SGD updater");
        }
        return new CustomGradientUpdater(this);
    }

    @Override
    public CustomIUpdater clone() {
        return new CustomIUpdater(learningRate);
    }

    @Override
    public double getLearningRate(int iteration, int epoch) {
        return learningRate;
    }

    @Override
    public boolean hasLearningRate() {
        return true;
    }

    @Override
    public void setLrAndSchedule(double lr, ISchedule iSchedule) {
        this.learningRate = lr;
    }
}
