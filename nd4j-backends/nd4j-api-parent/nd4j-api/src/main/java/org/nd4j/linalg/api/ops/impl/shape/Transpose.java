/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops.impl.shape;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.ShapeOp;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Collections;
import java.util.List;

/**
 * Transpose function
 *
 * @author Adam Gibson
 */
public class Transpose extends ShapeOp {
    public Transpose(SameDiff sameDiff, DifferentialFunction i_v) {
        super(sameDiff, i_v, ArrayUtil.reverseCopy(i_v.getShape()),false,null);
    }

    public Transpose() {}

    public Transpose(INDArray x, INDArray z) {
        super(x, z);
    }

    public Transpose(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public Transpose(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Transpose(INDArray x) {
        super(x);
    }


    @Override
    public void exec(int... dimensions) {
        exec();
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public void exec() {
        if(x != z) {
            z.assign(x.transpose());
        }
        else {
            this.z = x.transpose();
        }

    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "transpose";
    }

    @Override
    public String onnxName() {
        return "Transpose";
    }

    @Override
    public String tensorflowName() {
        return "transpose";
    }


    @Override
    public INDArray z() {
        if(x() != null)
            return x().transpose();
        return null;
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        return Collections.<DifferentialFunction>singletonList(this);
    }

}
