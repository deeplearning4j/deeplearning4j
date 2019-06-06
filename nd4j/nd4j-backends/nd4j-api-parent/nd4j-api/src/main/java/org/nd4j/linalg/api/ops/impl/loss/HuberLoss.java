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
import org.nd4j.base.Preconditions;

import java.util.Arrays;
import java.util.List;

/**
 * Huber loss
 *
 * @author Alex Black
 */
public class HuberLoss extends BaseLoss {

    private double delta;

    public HuberLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels, double delta){
        super(sameDiff, lossReduce, predictions, weights, labels);
        Preconditions.checkState(delta >= 0.0, "Delta must be >= 0.0. Got: %s", delta);
        this.delta = delta;
        tArguments.add(delta);
    }

    public HuberLoss(){ }

    @Override
    public String opName() {
        return "huber_loss";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        //No external gradient
        //Args are: predictions, weights, label
        SDVariable[] grads = f().lossHuberBp(arg(2), arg(0), arg(1), lossReduce, delta);
        return Arrays.asList(grads);
    }


}
