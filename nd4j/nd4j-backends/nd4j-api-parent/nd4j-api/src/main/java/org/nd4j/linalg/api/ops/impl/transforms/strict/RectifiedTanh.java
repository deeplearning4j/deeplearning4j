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

package org.nd4j.linalg.api.ops.impl.transforms.strict;

import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformStrictOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * RectifiedTanh
 *
 * Essentially max(0, tanh(x))
 *
 * @author raver119@gmail.com
 */
public class RectifiedTanh extends BaseTransformStrictOp {
    public RectifiedTanh(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public RectifiedTanh() {}

    public RectifiedTanh(INDArray x, INDArray z) {
        super(x, z);
    }

    public RectifiedTanh(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 38;
    }

    @Override
    public String opName() {
        return "rectified_tanh";
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
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(f().tanhRectifiedBp(arg(), f1.get(0)));
    }
}
