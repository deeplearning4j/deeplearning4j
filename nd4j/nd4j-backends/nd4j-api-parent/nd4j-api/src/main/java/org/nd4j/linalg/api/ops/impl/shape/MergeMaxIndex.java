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
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;


@NoArgsConstructor
public class MergeMaxIndex extends DynamicCustomOp {

    private DataType dataType = DataType.INT32;

    public MergeMaxIndex(@NonNull SameDiff sameDiff, @NonNull SDVariable... inputs) {
        super("mergemaxindex", sameDiff, inputs);
        addIArgument(dataType.toInt());

    }

    public MergeMaxIndex(@NonNull INDArray... inputs) {
        super("mergemaxindex", inputs, null);
        Preconditions.checkArgument(areEqualShapes(inputs), "All inputs have to be equal shapes");
        addIArgument(dataType.toInt());

    }

    public MergeMaxIndex(@NonNull SameDiff sd, @NonNull SDVariable[] x, @NonNull DataType dataType) {
        super("mergemaxindex", sd, x);
        this.dataType = dataType;
        addIArgument(dataType.toInt());
    }

    public MergeMaxIndex(@NonNull INDArray[] x, @NonNull DataType dataType) {
        super(x, null);
        Preconditions.checkArgument(areEqualShapes(x), "All inputs have to be equal shapes");
        this.dataType = dataType;
        addIArgument(dataType.toInt());

    }


    protected static boolean areEqualShapes(INDArray... inputs) {
        for (INDArray input : inputs) {
            if (!inputs[0].equalShapes(input)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public String opName() {
        return "mergemaxindex";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Collections.singletonList(this.dataType);
    }
}