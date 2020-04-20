/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.linalg.api.ops.impl.loss;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.loss.bp.LogLossBp;

import java.util.Arrays;
import java.util.List;

/**
 * Binary log loss, or cross entropy loss:
 * {@code -1/numExamples * sum_i (labels[i] * log(predictions[i] + epsilon) + (1-labels[i]) * log(1-predictions[i] + epsilon))}
 *
 * @author Alex Black
 */
public class LogLoss extends BaseLoss {
    public static final double DEFAULT_EPSILON = 1e-7;

    private double epsilon;

    public LogLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels, double epsilon){
        super(sameDiff, lossReduce, predictions, weights, labels);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    public LogLoss(SameDiff sameDiff, SDVariable labels, SDVariable predictions, SDVariable weights,
                   LossReduce lossReduce, double epsilon) {
        this(sameDiff, lossReduce, predictions, weights, labels, epsilon);
    }

    public LogLoss(INDArray labels, INDArray predictions, INDArray weights, LossReduce lossReduce, double epsilon){
        super(lossReduce, predictions, weights, labels);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }


    public LogLoss(){ }

    @Override
    public String opName() {
        return "log_loss";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        //No external gradient
        //Args are: predictions, weights, label
        return new LogLossBp(sameDiff, lossReduce, arg(0), arg(1), arg(2), epsilon).outputs();
    }

}
