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

package org.nd4j.linalg.api.ops.compat;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

public class CompatSparseToDense extends DynamicCustomOp {

    public CompatSparseToDense() {
        //
    }

    public CompatSparseToDense(INDArray indices, INDArray shape, INDArray values) {
        Preconditions.checkArgument(shape.isZ() && indices.isZ(), "Shape & indices arrays must have one integer data types");
        inputArguments.add(indices);
        inputArguments.add(shape);
        inputArguments.add(values);
    }



    public CompatSparseToDense(SameDiff sd, SDVariable indices, SDVariable shape, SDVariable values) {
        super(sd,new SDVariable[]{indices,shape,values});
    }

    public CompatSparseToDense(SameDiff sd, SDVariable indices, SDVariable shape, SDVariable values, SDVariable defaultValue) {
        super(sd,new SDVariable[]{indices,shape,values,defaultValue});
    }

    public CompatSparseToDense(INDArray indices, INDArray shape, INDArray values, INDArray defaultValue) {
        super(new INDArray[]{indices,shape,values,defaultValue},null);
    }

    @Override
    public List<DataBuffer> calculateOutputShape(OpContext oc) {
        return Arrays.asList(Nd4j.createBuffer(LongShapeDescriptor.fromShape(oc.getInputArrays().get(1).toLongVector(),oc.getInputArrays().get(0).dataType()).toShapeInfo()));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        if(!dArguments.isEmpty())
            return Arrays.asList(dataTypes.get(0));
        return Arrays.asList(dataTypes.get(0));
    }

    @Override
    public String opName() {
        return "compat_sparse_to_dense";
    }
}
