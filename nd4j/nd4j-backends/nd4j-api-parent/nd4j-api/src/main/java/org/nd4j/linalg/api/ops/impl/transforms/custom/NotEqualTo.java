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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Not equal to function:
 * Bit mask over whether 2 elements are not equal or not
 *
 * @author Adam Gibson
 */
public class NotEqualTo extends BaseDynamicTransformOp {
    public NotEqualTo() {}

    public NotEqualTo( SameDiff sameDiff, SDVariable x, SDVariable y) {
        this(sameDiff, new SDVariable[]{x,y}, false);
    }

    public NotEqualTo( SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(sameDiff, args, inPlace);
    }

    public NotEqualTo( INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }

    public NotEqualTo(INDArray x, INDArray y, INDArray z){
        this(new INDArray[]{x, y}, new INDArray[]{z});
    }

    public NotEqualTo(INDArray x, INDArray y){
        this(new INDArray[]{x, y}, null);
    }

    @Override
    public String opName() {
        return "not_equals";
    }

    @Override
    public String onnxName() {
        return "LogicalNot";
    }

    @Override
    public String tensorflowName() {
        return "NotEqual";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        //2 inputs, not continuously differentiable but 0s almost everywhere for gradient
        return Arrays.asList(sameDiff.zerosLike(args()[0]), sameDiff.zerosLike(args()[1]));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 2, "Expected exactly 2 input datatypes for %s, got %s", getClass(), dataTypes);
        Preconditions.checkState(dataTypes.get(0) == dataTypes.get(1), "Input datatypes must be same type: got %s", dataTypes);
        return Collections.singletonList(DataType.BOOL);
    }
}
