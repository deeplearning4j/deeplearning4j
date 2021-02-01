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

package org.nd4j.linalg.api.ops.impl.reduce.bp;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.List;


@NoArgsConstructor
public class DotBp extends BaseReductionBp {

    public DotBp(SameDiff sameDiff, SDVariable origInput1, SDVariable origInput2, SDVariable gradAtOutput, boolean keepDims, int... dimensions) {
        super(sameDiff, origInput1, origInput2, gradAtOutput, keepDims, dimensions);
    }

    public DotBp(INDArray origInput1, INDArray origInput2, INDArray gradAtOutput, INDArray output, boolean keepDims, int... dimensions){
        super(origInput1, origInput2, gradAtOutput, output, keepDims, dimensions);
    }

    public DotBp(INDArray origInput1, INDArray origInput2, INDArray gradAtOutput,
                 INDArray outputX, INDArray outputY, boolean keepDims, int... dimensions) {
        super(origInput1, origInput2, gradAtOutput, outputX, outputY, keepDims, dimensions);
    }

    @Override
    public String opName() {
        return "reduce_dot_bp";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 3, "Expected exactly 3 input datatype for %s, got input %s", getClass(), dataTypes);
        Preconditions.checkState(dataTypes.get(0).isFPType(), "First input must be a floating point type, got %s", dataTypes.get(0));
        Preconditions.checkState(dataTypes.get(1).isFPType(), "Second input (gradient at reduction output) must be a floating point type, got %s", dataTypes.get(1));
        Preconditions.checkState(dataTypes.get(2).isFPType(), "Second input (gradient at reduction output) must be a floating point type, got %s", dataTypes.get(2));
        return Arrays.asList(dataTypes.get(0), dataTypes.get(0));
    }
}
