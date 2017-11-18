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
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.Collections;
import java.util.List;

/**
 * ACosh elementwise function
 *
 * @author Adam Gibson
 */
public class ACosh extends BaseTransformOp {

    public ACosh() {}

    public ACosh(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public ACosh(INDArray x) {
        super(x);
    }

    public ACosh(INDArray x, INDArray y) {
        super(x, y);
    }

    public ACosh(INDArray indArray, INDArray indArray1, int length) {
        super(indArray, indArray1, length);
    }

    public ACosh(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace) {

        super(sameDiff, i_v, inPlace);
    }

    public ACosh(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public ACosh(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    @Override
    public int opNum() {
        return 16;
    }

    @Override
    public String opName() {
        return "acosh";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        return "acosh";
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        DifferentialFunction ret = f().div(f().one(getResultShape()),
                f().mul(f().sqrt(f().sub(arg(),f().one(getResultShape()))),f()
                        .sqrt(f().add(arg(),f().one(getResultShape())))));

        return Collections.singletonList(ret);
    }

}
