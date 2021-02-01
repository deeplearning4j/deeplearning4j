
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
package org.nd4j.linalg.api.ops.random.custom;

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
public class RandomShuffle extends DynamicCustomOp {

    public RandomShuffle(@NonNull INDArray value, int... seeds) {
        addInputArgument(value);
        addIArgument(seeds);
    }

    public RandomShuffle(@NonNull INDArray value) {
        this(value, 0, 0);
    }

    public RandomShuffle(@NonNull SameDiff sameDiff, @NonNull SDVariable value, int...seeds) {
        super(null, sameDiff, new SDVariable[]{value});
        addIArgument(seeds);
    }

    @Override
    public String opName() {
        return "random_shuffle";
    }

    @Override
    public String tensorflowName() {
        return "RandomShuffle";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 1, "Expected exactly 1 input datatype for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
