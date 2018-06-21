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

package org.nd4j.linalg.api.ops.random.impl;

import lombok.NonNull;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Inverted DropOut implementation as Op
 *
 * @author raver119@gmail.com
 */
public class DropOutInverted extends BaseRandomOp {

    private double p;

    public DropOutInverted() {
    }

    public DropOutInverted(SameDiff sameDiff, SDVariable input, double p) {
        super(sameDiff, input);
        this.p = p;
        //https://github.com/deeplearning4j/deeplearning4j/issues/5650
        throw new UnsupportedOperationException("Dropout SameDiff support disabled pending backprop support");
    }

    public DropOutInverted(@NonNull INDArray x, double p) {
        this(x, x, p, x.lengthLong());
    }

    public DropOutInverted(@NonNull INDArray x, @NonNull INDArray z, double p) {
        this(x, z, p, x.lengthLong());
    }

    public DropOutInverted(@NonNull INDArray x, @NonNull INDArray z, double p, long n) {
        this.p = p;
        init(x, null, z, n);
    }

    @Override
    public int opNum() {
        return 2;
    }

    @Override
    public String opName() {
        return "dropout_inverted";
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("p",p);
        return ret;
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[] {p};
    }

    @Override
    public String onnxName() {
        return "Dropout";
    }

    @Override
    public String tensorflowName() {
        return "Dropout";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }
}
