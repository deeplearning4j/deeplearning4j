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

package org.nd4j.linalg.api.ops.impl.loss;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;


/**
 * Softmax cross entropy loss with Logits
 *
 * @author Max Pumperla
 */
@NoArgsConstructor
public class SoftmaxCrossEntropyWithLogitsLoss extends DynamicCustomOp {

    protected int classesDim;

//    public SoftmaxCrossEntropyWithLogitsLoss(SameDiff sameDiff, SDVariable logits, SDVariable weights, SDVariable labels, int classesDim) {
//        super(null, sameDiff, new SDVariable[]{logits, weights, labels}, false);
//        this.classesDim = classesDim;
//        addIArgument(classesDim);
//    }

    public SoftmaxCrossEntropyWithLogitsLoss(SameDiff sameDiff, SDVariable logits, SDVariable labels, int classesDim) {
        super(null, sameDiff, new SDVariable[]{logits, labels}, false);
        this.classesDim = classesDim;
        addIArgument(classesDim);
    }

    @Override
    public String opName() {
        return "softmax_cross_entropy_loss_with_logits";
    }

    @Override
    public String tensorflowName() {
        return "SoftmaxCrossEntropyWithLogits";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && (inputDataTypes.size() == 2 || inputDataTypes.size() == 3),
                "Expected 2 or 3 input datatypes for %s, got %s", getClass(), inputDataTypes);

        return Collections.singletonList(inputDataTypes.get(0));    //Same as predictions
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        //No external gradient
        //Args: logits, weigths, label
        SDVariable[] args = args();
        SDVariable[] grads = f().lossSoftmaxCrossEntropyWithLogitsBp(arg(0), arg(1), classesDim);
        return Arrays.asList(grads);
    }
}
