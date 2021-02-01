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

public class BitsHammingDistance extends DynamicCustomOp {

    public BitsHammingDistance(){ }

    public BitsHammingDistance(@NonNull SameDiff sd, @NonNull SDVariable x, @NonNull SDVariable y){
        super(sd, new SDVariable[]{x, y});
    }

    public BitsHammingDistance(@NonNull INDArray x, @NonNull INDArray y){
        super(new INDArray[]{x, y}, null);
    }

    @Override
    public String opName() {
        return "bits_hamming_distance";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 2, "Expected 2 input datatypes, got %s", dataTypes);
        Preconditions.checkState(dataTypes.get(0).isIntType() && dataTypes.get(1).isIntType(), "Input datatypes must be integer type, got %s", dataTypes);
        return Collections.singletonList(DataType.LONG);
    }
}
