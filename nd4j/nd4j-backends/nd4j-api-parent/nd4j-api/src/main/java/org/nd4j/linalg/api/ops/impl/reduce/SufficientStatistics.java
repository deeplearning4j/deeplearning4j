/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

package org.nd4j.linalg.api.ops.impl.reduce;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.*;

/**
 * Sufficient statistics: returns 3 or 4 output arrays:
 * If shift is not provided: count, sum of elements, sum of squares
 * If shift is provided: count, sum of elements, sum of squares, shift
 *
 * @author Alex Black
 */
@NoArgsConstructor
public class SufficientStatistics extends DynamicCustomOp {

    public SufficientStatistics(SameDiff sameDiff, @NonNull SDVariable x, @NonNull SDVariable axis, SDVariable shift) {
        super(null, sameDiff, argsNoNull(x, axis, shift), false);
    }

    private static SDVariable[] argsNoNull(SDVariable x, SDVariable axis, SDVariable shift){
        if(shift == null){
            return new SDVariable[]{x, axis};
        } else {
            return new SDVariable[]{x, axis, shift};
        }
    }

    public SufficientStatistics(@NonNull INDArray x, @NonNull INDArray axes, INDArray shift) {
        if (shift != null)
            addInputArgument(x, axes, shift);
        else
            addInputArgument(x, axes);
    }

    public SufficientStatistics(@NonNull INDArray x, @NonNull INDArray axes) {
        this(x,axes,null);
    }

    @Override
    public String opName() {
        return "sufficient_statistics";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        throw new UnsupportedOperationException("Backprop not yet implemented for op: " + getClass().getSimpleName());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        // FIXME
        return Arrays.asList(inputDataTypes.get(0), inputDataTypes.get(0),inputDataTypes.get(0));
    }
}
