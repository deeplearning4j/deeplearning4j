/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.Op;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class TensorArrayConcat extends BaseTensorOp {

    public TensorArrayConcat(String name, SameDiff sameDiff, SDVariable[] args){
        super(name, sameDiff, args);
    }
    public TensorArrayConcat(SameDiff sameDiff, SDVariable[] args){
        super(null, sameDiff, args);
    }

    public TensorArrayConcat(){}
    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op name found for " + opName());
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]{"TensorArrayConcat", "TensorArrayConcatV2", "TensorArrayConcatV3"};
    }


    @Override
    public String toString() {
        return opName();
    }

    @Override
    public String opName() {
        return "stack_list";
    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(java.util.List<org.nd4j.linalg.api.buffer.DataType> inputDataType) {
        //Same output type as the TensorArray - which is defined by input 0
        SDVariable tArr = arg(0);
        TensorArray t3 = (TensorArray) sameDiff.getVariableOutputOp(tArr.name());
        org.nd4j.linalg.api.buffer.DataType dt = t3.getTensorArrayDataType();
        return Collections.singletonList(dt);
    }
}
