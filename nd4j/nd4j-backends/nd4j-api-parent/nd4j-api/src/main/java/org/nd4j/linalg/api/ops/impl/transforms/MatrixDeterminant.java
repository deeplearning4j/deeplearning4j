/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

/**
 * Matrix Determinant op
 *
 * Given input with shape [..., N, N] output the determinant for each sub-matrix.
 *
 * @author Alex Black
 */
public class MatrixDeterminant extends DynamicCustomOp {

    public MatrixDeterminant() {
        //
    }

    public MatrixDeterminant(SameDiff sameDiff, SDVariable in, boolean inPlace) {
        super(null, sameDiff, new SDVariable[]{in}, inPlace);
    }


    @Override
    public String opName() {
        return "matrix_determinant";
    }

    @Override
    public String tensorflowName() {
        return "MatrixDeterminant";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        //Derivative of matrix determinant
        //From: Matrix Cookbook - Petersen & Pedersen
        // z=det(X) then dz/dx = z * tr(X^-1)
        SDVariable transpose = f().matrixInverse(arg());
        SDVariable trace = f().diagPart(transpose).sum(-1);
        SDVariable dOutdIn = outputVariable().mul(trace);
        return Collections.singletonList(i_v.get(0).mul(dOutdIn));
    }
}
