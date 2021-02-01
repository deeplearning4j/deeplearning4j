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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

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
public class IsNonDecreasing extends DynamicCustomOp {

    public IsNonDecreasing( SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(null, sameDiff, args, inPlace);
    }

    public IsNonDecreasing( SameDiff sameDiff, SDVariable[] args) {
        super(null, sameDiff, args, false);
    }

    public IsNonDecreasing( SameDiff sameDiff, SDVariable input) {
        super(null, sameDiff, new SDVariable[]{input}, false);
    }

    public IsNonDecreasing(@NonNull INDArray input){
        this(input, null);
    }

    public IsNonDecreasing(@NonNull INDArray input, INDArray output) {
        super(null, new INDArray[]{input}, wrapOrNull(output));
    }


    @Override
    public String opName() {
        return "is_non_decreasing";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatype for %s, got %s", getClass(), dataTypes);
        return Collections.singletonList(DataType.BOOL);
    }
}
