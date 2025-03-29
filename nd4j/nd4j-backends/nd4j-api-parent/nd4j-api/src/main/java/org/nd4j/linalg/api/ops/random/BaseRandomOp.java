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

package org.nd4j.linalg.api.ops.random;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseOp;
import org.nd4j.linalg.api.ops.RandomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

@NoArgsConstructor
public abstract class BaseRandomOp extends BaseOp implements RandomOp {
    protected long[] shape;
    protected DataType dataType = Nd4j.defaultFloatingPointType();

    public BaseRandomOp(SameDiff sameDiff, SDVariable i_v) {
        Preconditions.checkNotNull(i_v, "Input variable can't be null with this constructor");
        this.sameDiff = sameDiff;
        this.xVertexId = i_v.name();
        if(i_v.getShape() != null)
            this.shape = i_v.getShape();
        else if(i_v.getArr() != null && i_v.getArr().shape() != null)
            this.shape = i_v.getArr().shape();
        sameDiff.addArgsFor(new String[]{xVertexId},this);
    }

    public BaseRandomOp(SameDiff sd, long[] shape){
        super(sd, null);
        Preconditions.checkArgument(shape != null && shape.length > 0, "Shape must be non-null, length > 0. Got: %s", shape);
        this.sameDiff = sd;
        this.shape = shape;
        setInstanceId();
        sameDiff.addArgsFor(new String[0], this);
    }

    public BaseRandomOp(INDArray x, INDArray y, INDArray z){
        super(x,y,z);
    }

    @Override
    public Type opType() {
        return Type.RANDOM;
    }

    @Override
    public List<DataBuffer> calculateOutputShape() {
        return calculateOutputShape(null);
    }



    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(DataType.FLOAT);
    }

    @Override
    public boolean isInPlace(){
        return x == null || x == z || x.data().pointer().address() == z.data().pointer().address();
    }

    public boolean isTripleArgRngOp(){
        return false;
    }
}
