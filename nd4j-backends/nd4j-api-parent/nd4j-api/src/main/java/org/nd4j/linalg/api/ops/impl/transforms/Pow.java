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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.Arrays;
import java.util.List;

/**
 * Pow function
 *
 * @author Adam Gibson
 */
public class Pow extends BaseTransformOp {
    private double pow;

    public Pow() {}

    public Pow(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace, double pow) {
        super(sameDiff, i_v, inPlace);
        this.pow = pow;
    }

    public Pow(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, boolean inPlace, Object[] extraArgs, double pow) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
        this.pow = pow;
    }

    public Pow(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs, double pow) {
        super(sameDiff, i_v, extraArgs);
        this.pow = pow;
    }

    public Pow(INDArray x, INDArray z, double pow) {
        super(x, z);
        this.pow = pow;
        init(x, null, z, x.lengthLong());
    }

    public Pow(INDArray x, INDArray z, long n, double pow) {
        super(x, z, n);
        this.pow = pow;
        init(x, null, z, n);

    }

    public Pow(INDArray x, INDArray y, INDArray z, long n, double pow) {
        super(x, y, z, n);
        this.pow = pow;
        init(x, y, z, n);

    }

    public Pow(INDArray x, double pow) {
        super(x);
        this.pow = pow;
        init(x, null, x, x.lengthLong());
    }

    @Override
    public int opNum() {
        return 7;
    }

    @Override
    public String opName() {
        return "pow";
    }


    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[] {pow};
    }


    @Override
    public String onnxName() {
        return "Pow";
    }

    @Override
    public String tensorflowName() {
        return "pow";
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v1) {
        DifferentialFunction g = f().mul(f().pow(arg(),scalarValue.doubleValue()),i_v1.get(0));

        return Arrays.asList(g);
    }

}
