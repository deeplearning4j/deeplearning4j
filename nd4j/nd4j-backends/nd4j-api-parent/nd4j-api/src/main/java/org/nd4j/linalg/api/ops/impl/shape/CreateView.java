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
package org.nd4j.linalg.api.ops.impl.shape;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;

public class CreateView extends DynamicCustomOp  {

    public CreateView(INDArray[] inputs) {
        super(inputs, null);
    }

    public CreateView(SameDiff sameDiff, SDVariable[] args) {
        super(sameDiff, args);
    }

    public CreateView(SameDiff sd, SDVariable input, SDVariable[] indices) {
        this(sd, ArrayUtil.combine(new SDVariable[]{input},indices));
    }

    public CreateView(INDArray input, INDArray[] indices) {
        this(ArrayUtil.combine(new INDArray[]{input},indices));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<org.nd4j.linalg.api.buffer.DataType> dataTypes) {
        Preconditions.checkState(dataTypes.size() == 1, "Expected list with exactly 1 datatype for %s, got %s", getClass(), dataTypes);
        //Output type is same as input type
        return dataTypes;
    }

    @Override
    public String opName() {
        return "create_view";
    }

}
