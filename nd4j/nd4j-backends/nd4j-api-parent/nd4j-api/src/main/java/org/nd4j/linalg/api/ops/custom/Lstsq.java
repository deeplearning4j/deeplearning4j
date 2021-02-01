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
public class Lstsq extends DynamicCustomOp {

    public Lstsq(@NonNull INDArray matrix, @NonNull INDArray rhs, double l2_regularizer, boolean fast) {
        addInputArgument(matrix, rhs);
        addTArgument(l2_regularizer);
        addBArgument(fast);
    }

    public Lstsq(@NonNull INDArray matrix, @NonNull INDArray rhs) {
        this(matrix, rhs, 0.0, true);
    }

    public Lstsq(@NonNull SameDiff sameDiff, @NonNull SDVariable matrix, @NonNull SDVariable rhs, double l2_regularizer, boolean fast) {
        super(sameDiff, new SDVariable[]{matrix,rhs});
        addTArgument(l2_regularizer);
        addBArgument(fast);
    }

    @Override
    public String opName() {
        return "lstsq";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        int n = args().length;
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == n, "Expected %s input data types for %s, got %s", n, getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
