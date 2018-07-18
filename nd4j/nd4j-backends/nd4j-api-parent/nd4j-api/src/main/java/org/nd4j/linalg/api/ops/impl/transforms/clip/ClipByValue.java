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

package org.nd4j.linalg.api.ops.impl.transforms.clip;

import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class ClipByValue extends DynamicCustomOp {

    private double clipValueMin;
    private double clipValueMax;

    public ClipByValue(INDArray[] inputs, INDArray[] outputs, double clipValueMin, double clipValueMax, boolean inPlace) {
        super(null, inputs, outputs);
        this.clipValueMin = clipValueMin;
        this.clipValueMax = clipValueMax;
        this.inplaceCall = inPlace;
        addTArgument(clipValueMin, clipValueMax);
    }

    public ClipByValue() {

    }

    public ClipByValue(SameDiff sameDiff, SDVariable x, double clipValueMin, double clipValueMax, boolean inPlace) {
        super(null, sameDiff, new SDVariable[]{x});
        this.clipValueMin = clipValueMin;
        this.clipValueMax = clipValueMax;
        this.inplaceCall = inPlace;
        addTArgument(clipValueMin, clipValueMax);
    }


    public ClipByValue(SameDiff sameDiff, SDVariable x, double clipValueMin, double clipValueMax) {
        super(null, sameDiff, new SDVariable[]{x});
        this.clipValueMin = clipValueMin;
        this.clipValueMax = clipValueMax;
        addTArgument(clipValueMin, clipValueMax);
    }

    @Override
    public String opName() {
        return "clipbyvalue";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        throw new UnsupportedOperationException("Not yet implemented");
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        //dOut/dIn is 0 if clipped, 1 otherwise
        SDVariable out = outputVariables()[0];
        SDVariable notClippedLower = f().gt(arg(), clipValueMin);
        SDVariable notClippedUpper = f().lt(arg(), clipValueMax);
        SDVariable ret = notClippedLower.mul(notClippedUpper).mul(grad.get(0));
        return Arrays.asList(ret);
    }
}
