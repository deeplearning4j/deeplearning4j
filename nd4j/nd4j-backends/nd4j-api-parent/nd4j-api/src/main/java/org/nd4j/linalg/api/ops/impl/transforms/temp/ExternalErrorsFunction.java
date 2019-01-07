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

package org.nd4j.linalg.api.ops.impl.transforms.temp;

import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

public class ExternalErrorsFunction extends DifferentialFunction {

    private static final List<LongShapeDescriptor> OUT_SHAPE = Collections.singletonList(LongShapeDescriptor.fromShape(new long[0], Nd4j.dataType()));

    private Map<String,INDArray> gradients;
    private Map<String,SDVariable> gradVariables;
    private SDVariable out;


    public ExternalErrorsFunction(SameDiff sd, List<SDVariable> inputs, Map<String,INDArray> gradients){
        super(sd, inputs.toArray(new SDVariable[inputs.size()]));
        if(gradients == null)
            gradients = new HashMap<>();
        this.gradients = gradients;
    }

    public ExternalErrorsFunction(){ }

    public void updateVariable(String str, INDArray gradient){
        gradients.put(str, gradient);
        //Update immediately if possible. New shapes might be needed for shape calculation
        if(gradVariables != null){
            gradVariables.get(str).setArray(gradient);
        }
    }

    @Override
    public SDVariable[] outputVariables(String baseName) {
        if(out == null){
            out = sameDiff.zero("dummyOutput", DataType.FLOAT, 1);
        }
        return new SDVariable[]{out};
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> out = new ArrayList<>();
        if (gradVariables == null) {
            gradVariables = new HashMap<>();
            for(SDVariable arg : args()){
                INDArray gradArr = gradients.get(arg.getVarName());
                SDVariable grad;
                if(gradArr != null){
                    grad = sameDiff.var(arg.getVarName() + "-externalGrad", gradArr);
                } else {
                    grad = sameDiff.var(arg.getVarName() + "-externalGrad", arg.dataType(), arg.getShape());
                }
                gradVariables.put(arg.getVarName(), grad);
                out.add(grad);
            }
        }
        return out;
    }


    public void updateBeforeExecution(){
        Preconditions.checkState(gradVariables != null, "Variables list is null - doDiff has not been called?");

        //Update external gradients ready for execution
        for(Map.Entry<String,SDVariable> e : gradVariables.entrySet()){
            INDArray extGradArray = gradients.get(e.getKey());
            if(extGradArray == null){
                throw new IllegalStateException("Cannot execute SameDiff instance with external errors: external gradient " +
                        "for variable " + e.getKey() + " has not been defined");
            }
            gradVariables.get(e.getKey()).setArray(extGradArray);
        }
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {

    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("Not supported: " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("Not supported: " +  opName());
    }

    @Override
    public String opName(){
        return "ExternalErrorsFn";
    }

    @Override
    public String toString(){
        return "ExternalErrorsFunction(" + (gradVariables != null ? gradVariables.keySet() : "") + ")";
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(){
        return OUT_SHAPE;
    }
}
