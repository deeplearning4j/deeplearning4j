/*-
 *
 * * Copyright 2015 Skymind,Inc. * * Licensed under the Apache License, Version 2.0 (the "License"); * you may not use
 * this file except in compliance with the License. * You may obtain a copy of the License at * *
 * http://www.apache.org/licenses/LICENSE-2.0 * * Unless required by applicable law or agreed to in writing, software *
 * distributed under the License is distributed on an "AS IS" BASIS, * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * either express or implied. * See the License for the specific language governing permissions and * limitations under
 * the License.
 *
 *
 */

package org.nd4j.linalg.api.ops.impl.scalar;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;

import java.util.List;

/**
 * Scalar floating-point remainder (fmod aka 'floormod')
 *
 * @author Adam Gibson
 */
public class ScalarFMod extends BaseScalarOp {
    public ScalarFMod() {}

    public ScalarFMod(INDArray x, INDArray y, INDArray z, long n, Number num) {
        super(x, y, z, n, num);
    }

    public ScalarFMod(INDArray x, Number num) {
        super(x, num);
    }

    public ScalarFMod(SameDiff sd, SDVariable in, Number number){
        super(sd, in, number);
    }

    @Override
    public int opNum() {
        return 18;
    }

    @Override
    public String opName() {
        return "fmod_scalar";
    }
    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        return i_v1;    //Return unmodified
    }
}
