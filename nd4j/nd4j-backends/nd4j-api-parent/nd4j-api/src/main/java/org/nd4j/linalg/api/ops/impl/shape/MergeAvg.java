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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.extern.slf4j.Slf4j;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Slf4j
public class MergeAvg extends DynamicCustomOp {

    public MergeAvg(SameDiff sameDiff, SDVariable... inputs) {
        super(null, sameDiff, inputs);
    }

    public MergeAvg(){ }

    @Override
    public String opName() {
        return "mergeavg";
    }


/*
    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        List<LongShapeDescriptor> ret = new ArrayList<>(1);
        ret.add(LongShapeDescriptor.fromShape(arg().getShape(), Nd4j.defaultFloatintPointType()));
        return ret;
    }
*/

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        // no-op
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        return "MergeAvg";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        int nArgs = args().length;
        SDVariable gradient = sameDiff.setupFunction(i_v.get(0)).div(nArgs);
        List<SDVariable> ret = new ArrayList<>();
        for (int i = 0; i < args().length; i++)
            ret.add(gradient);
        return ret;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        DataType first = dataTypes.get(0);
        for( int i=1; i<dataTypes.size(); i++ ){
            DataType dt = dataTypes.get(i);
            Preconditions.checkState(first == dt, "All inputs must have same datatype - got %s and %s for inputs 0 and %s respectively", first, dt, i);
        }
        //Output type is same as input types if FP, or default FP type otherwise
        if(first.isFPType()){
            return Collections.singletonList(first);
        } else {
            return Collections.singletonList(Nd4j.defaultFloatingPointType());
        }
    }

}
