/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.scalar;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.Relu6Derivative;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

public class Relu6 extends BaseScalarOp {
    public Relu6(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double cutoff) {
        super(sameDiff, i_v, cutoff, inPlace);
    }

    public Relu6(SameDiff sameDiff, SDVariable i_v, double cutoff) {
        this(sameDiff, i_v, false, cutoff);
    }

    public Relu6() {
        //
    }

    public Relu6(INDArray x, INDArray z, double cutoff) {
        super(x,null, z, cutoff);
    }
    public Relu6(INDArray x, double cutoff) {
        super(x, cutoff);
    }

    public Relu6(INDArray x, INDArray z) {
        super(x, null, z,0.0f);
    }


    public Relu6(INDArray x) {
        this(x, 0.0f);
    }

    @Override
    public int opNum() {
        return 40;
    }

    @Override
    public String opName() {
        return "relu6";
    }

    @Override
    public String onnxName() { throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        return "Relu6";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");

    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable dLdOut = i_v.get(0);
        return new Relu6Derivative(sameDiff, arg(), dLdOut, scalarValue.getDouble(0)).outputs();
    }
}
