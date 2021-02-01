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

package org.nd4j.linalg.api.ops.impl.loss;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.loss.bp.LogPoissonLossBp;

import java.util.List;

/**
 * Log Poisson loss
 *
 * Note: This expects that the input/predictions are log(x) not x!
 *
 * @author Paul Dubs
 */
public class LogPoissonLoss extends BaseLoss {
    private boolean full;

    public LogPoissonLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels){
        this(sameDiff, lossReduce, predictions, weights, labels, false);
    }

    public LogPoissonLoss(SameDiff sameDiff, SDVariable labels, SDVariable predictions, SDVariable weights,
                        LossReduce lossReduce, boolean full) {
        this(sameDiff, lossReduce, predictions, weights, labels, full);
    }

    public LogPoissonLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels, boolean full){
        super(sameDiff, lossReduce, predictions, weights, labels);
        this.full = full;
        addArgs();
    }

    public LogPoissonLoss(INDArray labels, INDArray predictions, INDArray weights, LossReduce lossReduce, boolean full){
        super(lossReduce, predictions, weights, labels);
        this.full = full;
        addArgs();
    }

    public LogPoissonLoss(){ }

    protected void addArgs(){
        super.addArgs();
        if(full){
            iArguments.add((long) 1);
        }
    }

    @Override
    public String opName() {
        return "log_poisson_loss";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        //No external gradient
        //Args are: predictions, weights, label
        return new LogPoissonLossBp(sameDiff, lossReduce, arg(0), arg(1), arg(2), full).outputs();
    }

}
