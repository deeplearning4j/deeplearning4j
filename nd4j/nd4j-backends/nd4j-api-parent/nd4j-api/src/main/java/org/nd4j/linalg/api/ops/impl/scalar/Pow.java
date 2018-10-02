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

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Pow function
 *
 * @author Adam Gibson
 */
public class Pow extends BaseTransformOp {
    private double pow;

    public Pow() {
    }

    public Pow(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double pow) {
        super(sameDiff, i_v, inPlace);
        this.pow = pow;
        this.extraArgs = new Object[]{pow};
    }

    public Pow(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, Object[] extraArgs, double pow) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
        this.pow = pow;
        this.extraArgs = new Object[]{pow};
    }

    public Pow(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs, double pow) {
        super(sameDiff, i_v, extraArgs);
        this.pow = pow;
        this.extraArgs = new Object[]{pow};
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
        this.extraArgs = new Object[]{pow};
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val weightsName = nodeDef.getInput(1);
        val variable = initWith.getVariable(weightsName);
        val tmp = initWith.getArrForVarName(weightsName);

        // if second argument is scalar - we should provide array of same shape
        if (tmp != null) {
            if (tmp.isScalar()) {
                this.pow = tmp.getDouble(0);
            }
        }
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
        SDVariable g = f().powDerivative(arg(), this.pow).mul(i_v1.get(0));
        return Arrays.asList(g);
    }

}
