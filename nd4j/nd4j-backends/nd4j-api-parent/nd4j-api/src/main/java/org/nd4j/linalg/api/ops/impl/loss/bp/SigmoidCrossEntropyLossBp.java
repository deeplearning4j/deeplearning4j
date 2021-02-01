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

package org.nd4j.linalg.api.ops.impl.loss.bp;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;


/**
 * Sigmoid cross entropy loss with logits
 *
 * @author Max Pumperla
 */
@NoArgsConstructor
public class SigmoidCrossEntropyLossBp extends BaseLossBp {
    private double labelSmoothing = 0.0;

    public SigmoidCrossEntropyLossBp(SameDiff sameDiff, LossReduce lossReduce, SDVariable logits, SDVariable weights,
                                     SDVariable labels, double labelSmoothing) {
        super(sameDiff, lossReduce, logits, weights, labels);
        this.labelSmoothing = labelSmoothing;
        addArgs();
    }

    public void addArgs() {
        super.addArgs();
        addTArgument(labelSmoothing);
    }

    @Override
    public String opName() {
        return "sigm_cross_entropy_loss_grad";
    }
}
