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
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

public class Reverse extends DynamicCustomOp {

    public Reverse(@NonNull SameDiff sameDiff, @NonNull SDVariable i_v, @NonNull int... dimensions) {
        super(sameDiff, new SDVariable[]{i_v});
        this.dimensions = dimensions;
        addIArgument(dimensions);
    }

    public Reverse() {
    }

    /**
     * Inplace reverse.  See {@link #Reverse(INDArray, INDArray)}
     */
    public Reverse(INDArray x){
        this(x, x);
        this.inPlace = true;
    }


    /**
     * This constructor allows to specify axis for Reverse operation
     * @param x
     * @param axis
     */
    public Reverse(INDArray x, int... axis){
        super(new INDArray[]{x}, new INDArray[0]);
        this.inPlace = false;
        this.dimensions = axis;
        addIArgument(axis);
    }

    /**
     * This constructor allows to specify axis for Reverse operation
     * @param x
     * @param axis
     */
    public Reverse(INDArray x, INDArray z, int... axis){
        super(new INDArray[]{x}, new INDArray[] {z});
        this.inPlace = false;
        this.dimensions = axis;
        addIArgument(axis);
    }

    /**
     * Reverses whole array for compatibility with OldReverse.
     *
     * Note that otherwise, passing null or empty dimensions will result in a noop.
     */
    public Reverse(INDArray x, INDArray z){
        super(new INDArray[]{x}, new INDArray[]{z});
        this.dimensions = new int[x.rank()];
        for(int i = 0 ; i < this.dimensions.length ; i++)
            this.dimensions[i] = i;
        addIArgument(dimensions);
    }

    @Override
    public String opName() {
        return "reverse";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "Reverse";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return new ReverseBp(sameDiff, arg(0), f1.get(0), dimensions).outputs();
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && (dataTypes.size() == 1 || dataTypes.size() == 2),
                "Expected 1 so 2 input datatype for %s, got %s", getClass(), dataTypes);
        return Collections.singletonList(dataTypes.get(0));
    }

}
