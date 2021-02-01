/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.loss.bp;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Binary log loss, or cross entropy loss:
 * {@code -1/numExamples * sum_i (labels[i] * log(predictions[i] + epsilon) + (1-labels[i]) * log(1-predictions[i] + epsilon))}
 *
 * @author Alex Black
 */
public class LogLossBp extends BaseLossBp {

    private double epsilon;

    public LogLossBp(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels, double epsilon){
        super(sameDiff, lossReduce, predictions, weights, labels);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    public LogLossBp(){ }

    @Override
    public String opName() {
        return "log_loss_grad";
    }


}
