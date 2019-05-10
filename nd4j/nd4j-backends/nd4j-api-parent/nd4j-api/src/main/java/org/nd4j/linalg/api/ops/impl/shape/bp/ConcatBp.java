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

package org.nd4j.linalg.api.ops.impl.shape.bp;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Backprop op for concat
 *
 * @author Alex Black
 */
@Slf4j
public class ConcatBp extends DynamicCustomOp {
    private int concatDimension;

    public ConcatBp(){

    }

    /**
     *
     * @param sameDiff
     * @param concatDimension
     * @param inputsAndGrad     Original inputs, followed by output gradient
     */
    public ConcatBp(SameDiff sameDiff, int concatDimension, SDVariable... inputsAndGrad){
        super(null, sameDiff, inputsAndGrad);
        addIArgument(concatDimension);
        this.concatDimension = concatDimension;
    }

    @Override
    public String opName() {
        return "concat_bp";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        //No op
    }


    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        //No op
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
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public int getNumOutputs(){
        return args().length - 1;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        SDVariable[] args = args();
        Preconditions.checkState(dataTypes.size() == args.length, "Expected list with exactly %s datatypes (original inputs + gradient), got %s", args.length, dataTypes);
        //Output type is same as (original) input types
        int n = getNumOutputs();
        List<DataType> out = new ArrayList<>(n);
        for( int i=0; i<n; i++){
            out.add(arg(i).dataType());
        }
        return out;
    }
}
