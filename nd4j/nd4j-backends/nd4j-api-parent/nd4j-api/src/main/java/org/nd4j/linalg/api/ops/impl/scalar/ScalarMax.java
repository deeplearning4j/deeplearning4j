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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;

import java.util.Collections;
import java.util.List;

/**
 * Scalar max operation.
 * Returns the max of an element
 * in the ndarray of the specified number.
 *
 * @author Adam Gibson
 */
public class ScalarMax extends BaseScalarOp {
    public ScalarMax() {}

    public ScalarMax(INDArray x, INDArray y, INDArray z, long n, Number num) {
        super(x, y, z, n, num);
    }

    public ScalarMax(INDArray x, Number num) {
        super(x, num);
    }


    public ScalarMax(SameDiff sd, SDVariable in, Number number){
        super(sd, in, number);
    }

    @Override
    public int opNum() {
        return 6;
    }

    @Override
    public String opName() {
        return "max_scalar";
    }

    @Override
    public String onnxName() {
        return "Max";
    }

    @Override
    public String tensorflowName() {
        return "RealMax";
    }


    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        if (scalarValue != null)
            this.extraArgs = new Object[]{scalarValue};

    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        SDVariable mask = arg().gt(scalarValue.doubleValue());
        return Collections.singletonList(i_v1.get(0).mul(mask));
    }
}
