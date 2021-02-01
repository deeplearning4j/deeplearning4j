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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * L2 loss op wrapper
 */
@NoArgsConstructor
public class L2Loss extends DynamicCustomOp {

    public L2Loss(SameDiff sameDiff, SDVariable var) {
        super(sameDiff, new SDVariable[]{var});
    }

    public L2Loss(INDArray var){
        super(new INDArray[]{var}, null);
    }

    @Override
    public String opName() {
        return "l2_loss";
    }

    @Override
    public String tensorflowName() {
        return "L2Loss";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 1, "Expected 1 input type for %s, got %s", getClass(), inputDataTypes);
        Preconditions.checkState(inputDataTypes.get(0).isFPType(), "Input datatype must be floating point for %s, got %s", getClass(), inputDataTypes);
        return inputDataTypes;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        //L2 loss: L = 1/2 * sum(x_i^2)
        //dL/dxi = xi
        return Collections.singletonList(sameDiff.identity(arg()));
    }
}
