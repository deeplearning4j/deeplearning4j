/* ******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.nd4j.linalg.api.ops.impl.transforms.clip;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;


@NoArgsConstructor
public class ClipByAvgNorm extends DynamicCustomOp {

    private double clipValue;


    public ClipByAvgNorm(SameDiff sameDiff, SDVariable x, double clipValue, int... dimensions) {
        super("clipbyavgnorm", sameDiff, new SDVariable[]{x});
        this.clipValue = clipValue;
        this.dimensions = dimensions;
        addIArgument(dimensions);
        addTArgument(clipValue);
    }

    public ClipByAvgNorm(INDArray in, double clipValue, int... dimensions){
        this(in, null, clipValue, dimensions);
    }

    public ClipByAvgNorm(INDArray in, INDArray out, double clipValue, int... dimensions){
        super("clipbyavgnorm", new INDArray[]{in}, wrapOrNull(out), Collections.singletonList(clipValue), dimensions);
    }

    @Override
    public String opName() {
        return "clipbyavgnorm";
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        throw new UnsupportedOperationException("Not yet implemented");    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 1, "Expected exactly 1 input datatype for %s, got %s", getClass(), inputDataTypes);
        return inputDataTypes;
    }

}


