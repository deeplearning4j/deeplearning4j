/*
 *  ******************************************************************************
 *  *
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

package org.nd4j.linalg.api.ops.impl.layers;

import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

public class ExternalErrorsFunction extends DynamicCustomOp {
    public static final String OP_NAME = "ExternalErrorsFn";

    private static final List<LongShapeDescriptor> OUT_SHAPE = Collections.singletonList(LongShapeDescriptor.fromShape(new long[0], Nd4j.dataType()));

    private Map<String,INDArray> gradients;
    private Map<String,SDVariable> gradVariables;
    private SDVariable out;
    private String id;


    public ExternalErrorsFunction(SameDiff sd, List<SDVariable> inputs, Map<String,INDArray> gradients){
        super(sd, inputs.toArray(new SDVariable[inputs.size()]));
        if(gradients == null)
            gradients = new HashMap<>();
        this.gradients = gradients;
        this.id = UUID.randomUUID().toString();
    }

    public ExternalErrorsFunction(){ }

    public String getGradPlaceholderName(){
        return arg().name() + "-grad";
    }

    @Override
    public SDVariable[] outputVariables(String baseName) {
        if(out == null){
            if(id == null)
                this.id = UUID.randomUUID().toString();
            String name = "dummyOutput-" + id;
            if(sameDiff.hasVariable(name)){
                out = sameDiff.getVariable(name);
            } else {
                out = sameDiff.zero(name, Nd4j.dataType(), 1);
                sameDiff.getOps().get(getOwnName()).setOutputsOfOp(Collections.singletonList(out.name()));
                sameDiff.getVariables().get(name).setOutputOfOp(getOwnName());
            }
        }
        return new SDVariable[]{out};
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> out = new ArrayList<>();
        if (gradVariables == null) {
            gradVariables = new HashMap<>();
            for(SDVariable arg : args()){
                INDArray gradArr = gradients.get(arg.name());
                SDVariable grad;
                DataType dt = arg.dataType();
                String n = getGradPlaceholderName();
                if(gradArr != null){
                    long[] shape = gradArr.shape().clone();
                    shape[0] = -1;
                    grad = sameDiff.var(n, VariableType.PLACEHOLDER, null, dt, shape);
                } else {
                    grad = sameDiff.var(n, VariableType.PLACEHOLDER, null, dt);
                }
                gradVariables.put(arg.name(), grad);
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
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {

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
        return OP_NAME;
    }

    @Override
    public String toString(){
        return "ExternalErrorsFunction(" + (gradVariables != null ? gradVariables.keySet() : "") + ")";
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(){
        return OUT_SHAPE;
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(OpContext oc){
        return OUT_SHAPE;
    }

    public Op.Type opType() {
        return Op.Type.LOGIC;
    }
}
