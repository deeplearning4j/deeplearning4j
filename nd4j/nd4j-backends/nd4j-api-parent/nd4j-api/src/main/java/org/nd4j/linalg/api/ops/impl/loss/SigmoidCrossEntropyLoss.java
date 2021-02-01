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

package org.nd4j.linalg.api.ops.impl.loss;

import lombok.NoArgsConstructor;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.loss.bp.SigmoidCrossEntropyLossBp;

import java.util.*;


/**
 * Sigmoid cross entropy loss with logits
 *
 * @author Max Pumperla
 */
@NoArgsConstructor
public class SigmoidCrossEntropyLoss extends BaseLoss {
    public static final double DEFAULT_LABEL_SMOOTHING = 0.0;
    private double labelSmoothing = 0.0;

    public SigmoidCrossEntropyLoss(SameDiff sameDiff, SDVariable labels, SDVariable logits, SDVariable weights,
                                    LossReduce lossReduce, double labelSmoothing) {
        this(sameDiff, lossReduce, logits, weights, labels, labelSmoothing);
    }

    public SigmoidCrossEntropyLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable logits, SDVariable weights,
                                   SDVariable labels, double labelSmoothing) {
        super(sameDiff, lossReduce, logits, weights, labels);
        this.labelSmoothing = labelSmoothing;
        addArgs();
    }

    public SigmoidCrossEntropyLoss(SameDiff sameDiff, LossReduce reductionMode, SDVariable logits, SDVariable weights, SDVariable labels) {
        this(sameDiff, reductionMode, logits, weights, labels, 0.0);
    }

    public SigmoidCrossEntropyLoss(INDArray labels, INDArray predictions, INDArray weights, LossReduce lossReduce, double labelSmoothing){
        super(lossReduce, predictions, weights, labels);
        this.labelSmoothing = labelSmoothing;
        addArgs();
    }

    public void addArgs() {
        super.addArgs();
        addTArgument(labelSmoothing);
    }

    @Override
    public String opName() {
        return "sigm_cross_entropy_loss";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        //No external gradient
        //Args are: predictions, weights, label
        return new SigmoidCrossEntropyLossBp(sameDiff, lossReduce, arg(0), arg(1), arg(2), labelSmoothing).outputs();
    }
}
