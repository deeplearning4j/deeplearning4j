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

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

public class MatrixDeterminant extends DynamicCustomOp {

    public MatrixDeterminant() {
        //
    }

    public MatrixDeterminant(@NonNull INDArray input){
        super(new INDArray[]{input}, null);
    }

    public MatrixDeterminant(SameDiff sameDiff, SDVariable in, boolean inPlace) {
        super(null, sameDiff, new SDVariable[]{in}, inPlace);
    }

    public MatrixDeterminant(SameDiff sameDiff, SDVariable in) {
       this(sameDiff, in, false);
    }


    @Override
    public String opName() {
        return "matrix_determinant";
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]{"MatrixDeterminant","BatchMatrixDeterminant"};
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        // Gradient of det(A): grad_A = upstream * det(A) * A^{-T}
        SDVariable input = arg();
        SDVariable grad = i_v.get(0);
        SDVariable det = outputVariable();
        // inv(A)^T * det(A) * upstream_grad
        SDVariable inv = sameDiff.math().matrixInverse(input);
        return Collections.singletonList(grad.mul(det).mul(sameDiff.transpose(inv)));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatype for %s, got %s", getClass(), dataTypes);
        Preconditions.checkState(dataTypes.get(0).isFPType(), "Input datatype must be a floating point type, got %s", dataTypes.get(0));
        return Collections.singletonList(dataTypes.get(0));
    }
}
