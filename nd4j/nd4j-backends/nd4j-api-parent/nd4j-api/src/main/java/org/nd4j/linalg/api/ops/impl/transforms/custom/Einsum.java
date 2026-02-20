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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Einsum (Einstein summation) operation.
 *
 * Einsum provides a powerful way to express tensor operations using Einstein summation notation.
 * The equation string specifies the subscripts for each input tensor and the output tensor.
 *
 * Examples:
 * - Matrix multiplication: "ij,jk->ik"
 * - Transpose: "ij->ji"
 * - Diagonal: "ii->i"
 * - Trace: "ii->"
 * - Batch matmul: "bij,bjk->bik"
 * - Dot product: "i,i->"
 * - Outer product: "i,j->ij"
 *
 * @author Adam Gibson
 */
public class Einsum extends DynamicCustomOp {

    private String equation;

    public Einsum() {
    }

    public Einsum(SameDiff sd, SDVariable[] inputs, String equation) {
        super(sd, inputs, false);
        this.equation = equation;
        addSArgument(equation);
    }

    public Einsum(INDArray[] inputs, String equation) {
        super(inputs, null);
        this.equation = equation;
        addSArgument(equation);
    }

    @Override
    public String opName() {
        return "einsum";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        throw new UnsupportedOperationException("Einsum gradient not yet implemented");
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // Output type is same as first input type
        if (dataTypes != null && !dataTypes.isEmpty()) {
            return Collections.singletonList(dataTypes.get(0));
        }
        return Collections.singletonList(DataType.FLOAT);
    }
}
