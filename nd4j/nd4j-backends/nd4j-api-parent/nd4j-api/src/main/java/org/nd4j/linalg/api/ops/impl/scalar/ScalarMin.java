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

import java.util.Collections;
import java.util.List;

/**
 * Scalar max operation.
 * Returns the max of an element
 * in the ndarray of the specified number.
 *
 * @author Adam Gibson
 */
public class ScalarMin extends BaseScalarOp {
    public ScalarMin() {}

    public ScalarMin(INDArray x, INDArray y, INDArray z, long n, Number num) {
        super(x, y, z, n, num);
    }

    public ScalarMin(INDArray x, Number num) {
        super(x, num);
    }

    public ScalarMin(SameDiff sd, SDVariable in, Number number){
        super(sd, in, number);
    }


    @Override
    public int opNum() {
        return 13;
    }

    @Override
    public String onnxName() {
        return "Min";
    }

    @Override
    public String tensorflowName() {
        return "RealMin";
    }


    @Override
    public String opName() {
        return "scalar_min";
    }



    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        if (scalarValue != null)
            this.extraArgs = new Object[] {scalarValue};


    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        SDVariable mask = arg().lt(scalarValue.getDouble(0)).castTo(arg().dataType());
        return Collections.singletonList(i_v1.get(0).mul(mask));
    }
}
