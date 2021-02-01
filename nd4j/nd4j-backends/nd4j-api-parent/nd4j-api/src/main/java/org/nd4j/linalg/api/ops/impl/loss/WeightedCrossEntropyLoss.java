/*
 *  ******************************************************************************
 *  *
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

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;

import java.util.Collections;
import java.util.List;


/**
 * Weighted cross entropy loss with logits
 *
 * @author Max Pumperla
 */
@NoArgsConstructor
public class WeightedCrossEntropyLoss extends DynamicCustomOp {


    public WeightedCrossEntropyLoss(SameDiff sameDiff, SDVariable targets, SDVariable inputs, SDVariable weights) {
        super(null, sameDiff, new SDVariable[]{targets, inputs, weights}, false);
        this.sameDiff = sameDiff;
    }

    public WeightedCrossEntropyLoss(@NonNull INDArray targets, @NonNull INDArray inputs, @NonNull INDArray weights){
        super(new INDArray[] {targets, inputs, weights}, null);
    }

    @Override
    public String opName() {
        return "weighted_cross_entropy_with_logits";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No TensorFlow op opName found for " + opName());
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 3, "Expected exactly 3 input datatypes for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));    //Same as predictions
    }
}
