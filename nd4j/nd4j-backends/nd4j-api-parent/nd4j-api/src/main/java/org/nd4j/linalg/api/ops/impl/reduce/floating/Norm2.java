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

package org.nd4j.linalg.api.ops.impl.reduce.floating;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceFloatOp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.Norm2Bp;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.List;

/**
 * Sum of squared values (real)
 * Sum of squared complex modulus (complex)
 *
 * @author Adam Gibson
 */
public class Norm2 extends BaseReduceFloatOp {
    public Norm2(SameDiff sameDiff, SDVariable i_v, boolean keepDims, int[] dimensions) {
        super(sameDiff, i_v, dimensions, keepDims);
    }

    public Norm2() {
    }

    public Norm2(INDArray x, INDArray z, int... dimensions) {
        super(x, null, z, dimensions);
    }

    public Norm2(INDArray x, int... dimensions) {
        super(x, dimensions);
    }

    public Norm2(INDArray x, boolean keepDims, int... dimensions) {
        super(x, keepDims, dimensions);
    }

    @Override
    public INDArray noOp() {
        return Transforms.abs(x());
    }


    @Override
    public int opNum() {
        return 3;
    }

    @Override
    public String opName() {
        return "reduce_norm2";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        //d norm2(in)/dx = x / norm2(in)
        return new Norm2Bp(sameDiff, arg(), grad.get(0), keepDims, dimensions).outputs();
    }


    @Override
    public String onnxName() {
        return "Norm";
    }

}
