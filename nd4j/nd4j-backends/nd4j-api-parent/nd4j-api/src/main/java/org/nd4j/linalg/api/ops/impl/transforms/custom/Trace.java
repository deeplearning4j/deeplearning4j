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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Matrix trace operation
 *
 * @author Alex Black
 */
public class Trace extends DynamicCustomOp {

    public Trace(SameDiff sd, SDVariable in){
        super(null, sd, new SDVariable[]{in});
    }

    public Trace(@NonNull INDArray in){
        super(wrapOrNull(in), null);
    }

    public Trace(){ }

    @Override
    public String opName(){
        return "trace";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradAtOutput){
        SDVariable rows = sameDiff.reshape(sameDiff.sizeAt(arg(), -2), 1);
        SDVariable cols = sameDiff.reshape(sameDiff.sizeAt(arg(), -1), 1);
        SDVariable eye = sameDiff.math().eye(/*sameDiff.shape(gradAtOutput.get(0)),*/ rows, cols);
        //Reshape gradient from [x,y,z] to [x,y,z,1,1]
        SDVariable reshapedGrad = sameDiff.expandDims(gradAtOutput.get(0), -1);
        reshapedGrad = sameDiff.expandDims(reshapedGrad, -1);
        return Collections.singletonList(reshapedGrad.mul(eye));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatype for %s, got %s", getClass(), dataTypes);
        return Collections.singletonList(dataTypes.get(0));
    }

}
