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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;


/**
 * Pairwise cross-product of two tensors of the same shape.
 *
 * Base operation for two vectors is:
 *  a x b = (a_2 * b_3 - a_3 * b_2, a_3 * b_1 - a_1 * b_3, a_1 * b_2 - a_2 * b_1)
 *
 * For tensors of more complicated shape this op is computed pairwise. For this
 * to work the outer dimension has to be 3.
 *
 * @author Max Pumperla
 */
@NoArgsConstructor
public class Cross extends DynamicCustomOp {

    public Cross(SameDiff sameDiff, SDVariable[] args) {
        super(null, sameDiff, args, false);
    }

    public Cross(SameDiff sameDiff, SDVariable a, SDVariable b) {
        this(sameDiff, new SDVariable[]{a,b});
    }

    public Cross(INDArray a, INDArray b){
        this(a,b,null);
    }

    public Cross(INDArray a, INDArray b, INDArray out){
        super(null, new INDArray[]{a,b}, wrapOrNull(out), null, (int[])null);
    }

    @Override
    public String opName() {
        return "cross";
    }


    @Override
    public String tensorflowName() {
        return "Cross";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        /**
         * dL / dx = dL / dCross * dCross / dx
         * dCross(a,b) / da = Cross(1, b)
         * dCross(a,b) / db = Cross(a, 1)
         *
         * return (grad * Cross(1, b), grad * Cross(a, 1)
         */
        SDVariable grad = gradients.get(0);
        SDVariable a = larg();
        SDVariable b = rarg();
        SDVariable ones = sameDiff.onesLike(a);

        SDVariable gradLeft = grad.mul(sameDiff.math().cross(b, ones));
        SDVariable gradRight = grad.mul(sameDiff.math().cross(ones, a));

        return Arrays.asList(gradLeft, gradRight);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes.size() == 2, "Expected list with exactly 2 datatype for %s, got %s", getClass(), dataTypes);
        //Output type is same as input type
        return Collections.singletonList(dataTypes.get(0));
    }
}
