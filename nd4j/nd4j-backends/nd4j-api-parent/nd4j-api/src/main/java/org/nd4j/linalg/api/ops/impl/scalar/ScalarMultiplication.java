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

package org.nd4j.linalg.api.ops.impl.scalar;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Scalar multiplication
 *
 * @author Adam Gibson
 */
public class ScalarMultiplication extends BaseScalarOp {

    public ScalarMultiplication() {}

    public ScalarMultiplication(INDArray x, INDArray y, INDArray z, long n, Number num) {
        super(x, y, z, n, num);
    }

    public ScalarMultiplication(INDArray x, Number num) {
        super(x, num);
    }


    public ScalarMultiplication(INDArray x) {
        this(x, 0);
    }


    public ScalarMultiplication(SameDiff sameDiff, SDVariable i_v, Number scalar) {
        super(sameDiff, i_v, scalar);
    }

    public ScalarMultiplication(SameDiff sameDiff, SDVariable i_v, Number scalar, boolean inPlace) {
        super(sameDiff, i_v, scalar, inPlace);
    }

    public ScalarMultiplication(SameDiff sameDiff, SDVariable i_v, Number scalar, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, scalar, inPlace, extraArgs);
    }

    public ScalarMultiplication(SameDiff sameDiff, SDVariable i_v, Number scalar, Object[] extraArgs) {
        super(sameDiff, i_v, scalar, extraArgs);
    }

    @Override
    public int opNum() {
        return 2;
    }

    @Override
    public String opName() {
        return "mul_scalar";
    }
    @Override
    public String onnxName() {
        return "Mul";
    }

    @Override
    public String tensorflowName() {
        return "RealMul";
    }




    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        return Collections.singletonList(i_v1.get(0).mul(scalarValue.doubleValue()));
    }
}
