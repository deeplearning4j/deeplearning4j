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

package org.nd4j.linalg.api.ops.impl.scalar;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;

import java.util.Collections;
import java.util.List;

/**
 * Pow function
 *
 * @author Adam Gibson
 */
public class Pow extends BaseScalarOp {
    private double pow;

    public Pow() {
    }

    public Pow(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double pow) {
        super(sameDiff, i_v, pow, inPlace);
        this.pow = pow;
        this.extraArgs = new Object[]{pow};
    }

    public Pow(SameDiff sameDiff, SDVariable i_v, double pow) {
        this(sameDiff, i_v, false, pow);
    }


    public Pow(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs, double pow) {
        super(sameDiff, i_v, pow, extraArgs);
        this.pow = pow;
        this.extraArgs = new Object[]{pow};
    }

    public Pow(INDArray x, INDArray z, double pow) {
        super(x, z, pow);
        this.pow = pow;
        this.extraArgs = new Object[]{pow};
    }

    public Pow(INDArray x, double pow) {
        super(x, pow);
        this.pow = pow;
        this.extraArgs = new Object[]{pow};
    }

    @Override
    public int opNum() {
        return 31;
    }

    @Override
    public String opName() {
        return "pow";
    }

    @Override
    public String onnxName() {
        return "Pow";
    }

    @Override
    public String tensorflowName() {
        return "Pow";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        SDVariable g = new PowDerivative(sameDiff, arg(), false, this.pow).outputVariable().mul(i_v1.get(0));
        return Collections.singletonList(g);
    }
}
