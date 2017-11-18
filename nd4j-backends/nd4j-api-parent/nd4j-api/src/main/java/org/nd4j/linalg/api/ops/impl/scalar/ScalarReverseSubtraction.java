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

package org.nd4j.linalg.api.ops.impl.scalar;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;

import java.util.Arrays;
import java.util.List;

/**
 * Scalar reverse subtraction
 *
 * @author Adam Gibson
 */
public class ScalarReverseSubtraction extends BaseScalarOp {

    public ScalarReverseSubtraction() {}

    public ScalarReverseSubtraction(INDArray x, INDArray y, INDArray z, long n, Number num) {
        super(x, y, z, n, num);
    }

    public ScalarReverseSubtraction(INDArray x, Number num) {
        super(x, num);
    }

    public ScalarReverseSubtraction(INDArray x, INDArray y, INDArray z, long n, IComplexNumber num) {
        super(x, y, z, n, num);
    }

    public ScalarReverseSubtraction(INDArray x, IComplexNumber num) {
        super(x, num);
    }

    public ScalarReverseSubtraction(SameDiff sameDiff, DifferentialFunction i_v, Number scalar) {
        super(sameDiff, i_v, scalar);
    }

    public ScalarReverseSubtraction(SameDiff sameDiff, DifferentialFunction i_v, Number scalar, boolean inPlace) {
        super(sameDiff, i_v, scalar, inPlace);
    }

    public ScalarReverseSubtraction(SameDiff sameDiff, DifferentialFunction i_v, Number scalar, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, scalar, inPlace, extraArgs);
    }

    public ScalarReverseSubtraction(SameDiff sameDiff, DifferentialFunction i_v, Number scalar, Object[] extraArgs) {
        super(sameDiff, i_v, scalar, extraArgs);
    }

    @Override
    public int opNum() {
        return 5;
    }

    @Override
    public String opName() {
        return "rsub_scalar";
    }


    @Override
    public String onnxName() {
        return "Sub";
    }

    @Override
    public String tensorflowName() {
        return "sub";
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v1) {
        DifferentialFunction g = f().neg(i_v1.get(0));

        return Arrays.asList(g);
    }

}
