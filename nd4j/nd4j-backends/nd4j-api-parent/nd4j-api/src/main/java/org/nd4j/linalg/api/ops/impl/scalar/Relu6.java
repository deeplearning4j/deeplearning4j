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
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Rectified linear unit 6, i.e. min(max(input, cutoff), 6), where cutoff can be chosen.
 *
 * @author Max Pumperla
 */
public class Relu6 extends BaseScalarOp {
    public Relu6(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double cutoff) {
        super(sameDiff, i_v, cutoff, inPlace);
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
        //TF cutoff is always 0.0. Need to make sure scalar type is same as input type (due to scalar op 'same type' exec restrictions)
        if(attributesForNode.containsKey("T")){
            attributesForNode.get("T").getType();
            DataType dt = TFGraphMapper.convertType(attributesForNode.get("T").getType());
            scalarValue = Nd4j.scalar(dt, 0.0);
        }
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable dLdOut = i_v.get(0);
        return Collections.singletonList(f().relu6Derivative(arg(), dLdOut, scalarValue.getDouble(0)));
    }
}
