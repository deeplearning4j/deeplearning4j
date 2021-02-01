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

package org.nd4j.linalg.api.ops.impl.shape.tensorops;

import onnx.Onnx;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class TensorArrayRead extends BaseTensorOp {

    protected DataType importDataType;

    public TensorArrayRead(String name, SameDiff sameDiff, SDVariable[] args){
        super(name, sameDiff, args);
    }
    public TensorArrayRead(SameDiff sameDiff, SDVariable[] args){
        super(null, sameDiff, args);
    }

    public TensorArrayRead(){}

    @Override
    public String[] tensorflowNames() {
        return new String[]{"TensorArrayRead", "TensorArrayReadV2", "TensorArrayReadV3"};
    }


    @Override
    public String opName() {
        return "read_list";
    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);

        this.importDataType = TFGraphMapper.convertType(attributesForNode.get("dtype").getType());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataType) {
        //Same output type as the TensorArray - which is defined by input 0
        DataType dt = null;
        if(importDataType != null) {
            dt = importDataType;
        } else {
            for(int i = 0; i < args().length; i++) {
                SDVariable tArr = arg(i);
                DifferentialFunction op = sameDiff.getVariableOutputOp(tArr.name());
                if(op instanceof TensorArray) {
                    TensorArray t3 = (TensorArray) op;
                    dt = t3.getTensorArrayDataType();
                    break;
                }

            }

        }
        return Collections.singletonList(dt);
    }
}
