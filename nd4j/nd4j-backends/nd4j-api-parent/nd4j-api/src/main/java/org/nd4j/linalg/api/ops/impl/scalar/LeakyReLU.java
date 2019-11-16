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

import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

/**
 * Leaky Rectified linear unit. Default alpha=0.01, cutoff=0<br>
 * Out(x) = alpha*x if x<0<br>
 * Out(x) = x if x >= 0<br>
 * Leaky ReLU may avoid zero gradient "dying ReLU" problem by having non-zero
 * gradient below 0.<br>
 * See for example https://arxiv.org/abs/1505.00853 for a comparison of
 * ReLU variants.
 *
 * @author Alex Black
 */
public class LeakyReLU extends BaseScalarOp {
    public static final double DEFAULT_ALPHA = 0.01;
    private double alpha = DEFAULT_ALPHA;

    public LeakyReLU(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double alpha) {
        super(sameDiff, i_v, alpha, inPlace);
        this.alpha = alpha;
        this.extraArgs = new Object[]{alpha};

    }

    public LeakyReLU(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs, double alpha) {
        super(sameDiff, i_v, alpha, extraArgs);
        this.alpha = alpha;
        this.extraArgs = new Object[]{alpha};
    }

    public LeakyReLU() {
        super();
    }

    public LeakyReLU(INDArray x, double alpha) {
        super(x, alpha);
        this.alpha = alpha;
        this.extraArgs = new Object[]{alpha};
    }

    public LeakyReLU(INDArray x, INDArray z, double alpha) {
        super(x, null, z, alpha);
        this.alpha = alpha;
        this.extraArgs = new Object[]{alpha};
    }


    public LeakyReLU(INDArray x, INDArray z) {
        this(x, z, 0.01);
    }

    public LeakyReLU(INDArray x) {
        super(x, 0.01);
    }

    @Override
    public int opNum() {
        return 35;
    }

    @Override
    public String opName() {
        return "leakyrelu";
    }

    @Override
    public String onnxName() {
        return "LeakyRelu";
    }

    @Override
    public String tensorflowName() {
        return "LeakyRelu";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Collections.singletonList(f().leakyReluBp(arg(), i_v.get(0), alpha));
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode,
            GraphDef graph) {
        alpha = attributesForNode.get("alpha").getF();
        extraArgs = new Object[]{alpha};
        this.setScalar(Nd4j.scalar(org.nd4j.linalg.api.buffer.DataType.FLOAT, alpha));
    }
}
