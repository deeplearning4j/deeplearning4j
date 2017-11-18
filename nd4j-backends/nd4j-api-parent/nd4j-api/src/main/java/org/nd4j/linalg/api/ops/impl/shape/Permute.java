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

import java.util.Collections;
import java.util.List;

/**
 * Permute function
 *
 * @author Adam Gibson
 */
public class Permute extends ShapeOp {
    private int[] permuteDims;

    public Permute(SameDiff sameDiff, DifferentialFunction i_v, int[] permuteDims) {
        super(sameDiff, i_v, false);
        this.permuteDims = permuteDims;
    }

    public Permute() {}

    public Permute(INDArray x, INDArray z,Object[] extraArgs) {
        super(x, z);
    }

    public Permute(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public Permute(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Permute(INDArray x) {
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
            z.assign(x.permute(permuteDims));
        }
        else {
            x.permutei(permuteDims);
        }

    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "permute";
    }

      @Override
    public INDArray z() {
        return x().transpose();
    }





    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        return Collections.<DifferentialFunction>singletonList(this);
    }

    @Override
    public String onnxName() {
        return "transpose";
    }

    @Override
    public String tensorflowName() {
        return "transpose";
    }


}
