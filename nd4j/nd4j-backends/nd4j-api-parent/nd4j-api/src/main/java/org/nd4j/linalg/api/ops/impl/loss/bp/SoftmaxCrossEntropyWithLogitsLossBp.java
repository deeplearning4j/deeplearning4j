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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;


@NoArgsConstructor
public class SoftmaxCrossEntropyWithLogitsLossBp extends DynamicCustomOp {

    protected int classesDim;

    public SoftmaxCrossEntropyWithLogitsLossBp(SameDiff sameDiff, SDVariable logits, SDVariable labels, int classesDim) {
        super(null, sameDiff, new SDVariable[]{logits, labels}, false);
        this.classesDim = classesDim;
        addIArgument(classesDim);
    }

    @Override
    public String opName() {
        return "softmax_cross_entropy_loss_with_logits_grad";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        throw new UnsupportedOperationException("Differentiation of " + getClass().getName() + " not supported");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Arrays.asList(arg(0).dataType(), arg(1).dataType());
    }
}
