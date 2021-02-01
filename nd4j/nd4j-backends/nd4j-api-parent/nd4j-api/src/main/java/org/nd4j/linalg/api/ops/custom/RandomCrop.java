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
package org.nd4j.linalg.api.ops.custom;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

public class RandomCrop extends DynamicCustomOp {

    public RandomCrop() {}

    public RandomCrop(@NonNull INDArray input, @NonNull INDArray shape) {
        Preconditions.checkArgument(shape.isVector(),"RandomCrop:Shape tensor should be a vector");
        Preconditions.checkArgument(input.rank() == shape.length(), "RandomCrop:The length of the shape vector is not match input rank");
        addInputArgument(input, shape);
    }

    public RandomCrop(@NonNull SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable shape) {
            super("", sameDiff, new SDVariable[]{input, shape});
    }

    @Override
    public String opName() {
        return "random_crop";
    }

    @Override
    public String tensorflowName() {
        return "RandomCrop";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null /*&& inputDataTypes.size() == 4*/,
                "Expected 4 input datatypes for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(DataType.FLOAT);   //TF import: always returns float32...
    }
}
