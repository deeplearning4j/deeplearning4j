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

package org.nd4j.linalg.api.ops;

import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class NoOp extends DynamicCustomOp {

    public NoOp(){ }

    public NoOp(SameDiff sd, SDVariable in){
        super("noop", sd, new SDVariable[]{in});
    }

    public NoOp(INDArray in) {
        addInputArgument(in);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(f1.get(0));
    }



    @Override
    public String opName() {
        return "noop";
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {

    }

    @Override
    public String onnxName() {
        return "NoOp";
    }

    @Override
    public String tensorflowName() {
        return "NoOp";
    }

    @Override
    public int getNumOutputs(){
        return 1;
    }


    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        if(inputArguments != null && !inputArguments.isEmpty()){
            return Collections.singletonList(inputArguments.get(0).shapeDescriptor());
        }
        return Collections.singletonList(Nd4j.empty(DataType.BOOL).shapeDescriptor());
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(OpContext oc){
        if(oc.getInputArrays() != null && !oc.getInputArrays().isEmpty()){
            return Collections.singletonList(oc.getInputArray(0).shapeDescriptor());
        }
        return Collections.singletonList(Nd4j.empty(DataType.BOOL).shapeDescriptor());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        return Collections.singletonList(DataType.BOOL);
    }
}
