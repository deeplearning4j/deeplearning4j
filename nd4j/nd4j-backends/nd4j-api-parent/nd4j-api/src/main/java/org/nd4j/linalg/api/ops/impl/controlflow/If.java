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

package org.nd4j.linalg.api.ops.impl.controlflow;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.CustomOpDescriptor;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.util.HashUtil;
import org.nd4j.weightinit.impl.ZeroInitScheme;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Equivalent to tensorflow's conditional op.
 * Runs one of 2 {@link SameDiff.SameDiffFunctionDefinition}
 * depending on a predicate {@link org.nd4j.autodiff.samediff.SameDiff.SameDiffConditional}
 *
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
@Slf4j
public class If extends DifferentialFunction implements CustomOp {

    @Getter
    protected SameDiff loopBodyExecution,predicateExecution,falseBodyExecution;


    @Getter
    protected SameDiff.SameDiffConditional predicate;
    @Getter
    protected SameDiff.SameDiffFunctionDefinition trueBody,falseBody;

    @Getter
    protected String blockName,trueBodyName,falseBodyName;

    @Getter
    protected SDVariable[] inputVars;

    @Getter
    protected Boolean trueBodyExecuted = null;

    @Getter
    protected SDVariable targetBoolean;

    protected SDVariable dummyResult;

    @Getter
    @Setter
    protected SDVariable[] outputVars;

    public If(If ifStatement) {
        this.sameDiff = ifStatement.sameDiff;
        this.outputVars = ifStatement.outputVars;
        this.falseBodyExecution = ifStatement.falseBodyExecution;
        this.trueBodyExecuted = ifStatement.trueBodyExecuted;
        this.falseBody = ifStatement.falseBody;
        this.trueBodyExecuted = ifStatement.trueBodyExecuted;
        this.dummyResult = ifStatement.dummyResult;
        this.inputVars = ifStatement.inputVars;
        this.dummyResult =  this.sameDiff.var("dummyresult-" + UUID.randomUUID().toString(),new long[]{1,1},new ZeroInitScheme());
        if(sameDiff.getShapeForVarName(dummyResult.getVarName()) == null)
            sameDiff.putShapeForVarName(dummyResult.getVarName(),new long[]{1,1});




    }

    @Builder
    public If(String blockName,
              SameDiff parent,
              SDVariable[] inputVars,
              SameDiff.SameDiffFunctionDefinition conditionBody,
              SameDiff.SameDiffConditional predicate,
              SameDiff.SameDiffFunctionDefinition trueBody,
              SameDiff.SameDiffFunctionDefinition falseBody) {

        this.sameDiff = parent;
        parent.putFunctionForId(getOwnName(),this);
        this.inputVars = inputVars;
        this.predicate = predicate;

        parent.addArgsFor(inputVars,this);
        this.trueBody = trueBody;
        this.falseBody = falseBody;
        this.blockName = blockName;
        //need to add the op to the list of ops to be executed when running backwards
        this.dummyResult =  parent.var("dummyresult-" + UUID.randomUUID().toString(),new long[]{1,1},new ZeroInitScheme('f'));
        parent.addOutgoingFor(new SDVariable[]{dummyResult},this);

        //create a samediff sub graph for running just the execution
        //return a reference to the loop for referencing during actual execution
        SameDiff sameDiff = SameDiff.create();
        //store the reference to the result array and the same diff execution instance
        this.targetBoolean = predicate.eval(sameDiff,conditionBody, inputVars);
        this.predicateExecution = sameDiff;
        //store references to the loop body
        String trueBodyName = "true-body-" + UUID.randomUUID().toString();
        this.trueBodyName = trueBodyName;

        String falseBodyName = "false-body-" + UUID.randomUUID().toString();
        this.falseBodyName = trueBodyName;

        //running define function will setup a proper same diff instance
        this.loopBodyExecution = parent.defineFunction(trueBodyName,trueBody,inputVars);
        this.falseBodyExecution = parent.defineFunction(falseBodyName,falseBody,inputVars);
        parent.defineFunction(blockName,conditionBody,inputVars);
        parent.putSubFunction("predicate-eval-body-" + UUID.randomUUID().toString(),sameDiff);
        //get a reference to the actual loop body
        this.loopBodyExecution = parent.getFunction(trueBodyName);
    }


    /**
     * Toggle whether the true body was executed
     * or the false body
     * @param trueBodyExecuted
     */
    public void exectedTrueOrFalse(boolean trueBodyExecuted)  {
        if(trueBodyExecuted)
            this.trueBodyExecuted = true;
        else
            this.trueBodyExecuted = false;
    }



    @Override
    public SDVariable[] outputVariables(String baseName) {
        return new SDVariable[]{dummyResult};
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> ret = new ArrayList<>();
        ret.addAll(Arrays.asList(new IfDerivative(this).outputVariables()));
        return ret;
    }

    @Override
    public String toString() {
        return opName();
    }

    @Override
    public String opName() {
        return "if";
    }

    @Override
    public long opHash() {
        return HashUtil.getLongHash(opName());
    }

    @Override
    public boolean isInplaceCall() {
        return false;
    }

    @Override
    public INDArray[] outputArguments() {
        return new INDArray[0];
    }

    @Override
    public INDArray[] inputArguments() {
        return new INDArray[0];
    }

    @Override
    public long[] iArgs() {
        return new long[0];
    }

    @Override
    public double[] tArgs() {
        return new double[0];
    }

    @Override
    public void addIArgument(int... arg) {

    }

    @Override
    public void addIArgument(long... arg) {

    }

    @Override
    public void removeIArgument(Integer arg) {

    }

    @Override
    public Long getIArgument(int index) {
        return null;
    }

    @Override
    public int numIArguments() {
        return 0;
    }

    @Override
    public void addTArgument(double... arg) {

    }

    @Override
    public void removeTArgument(Double arg) {

    }

    @Override
    public Double getTArgument(int index) {
        return null;
    }

    @Override
    public int numTArguments() {
        return 0;
    }

    @Override
    public void addInputArgument(INDArray... arg) {

    }

    @Override
    public void removeInputArgument(INDArray arg) {

    }

    @Override
    public INDArray getInputArgument(int index) {
        return null;
    }

    @Override
    public int numInputArguments() {
        return 0;
    }

    @Override
    public void addOutputArgument(INDArray... arg) {

    }

    @Override
    public void removeOutputArgument(INDArray arg) {

    }

    @Override
    public INDArray getOutputArgument(int index) {
        return null;
    }

    @Override
    public int numOutputArguments() {
        return 0;
    }

    @Override
    public Op.Type opType() {
        return  Op.Type.CONDITIONAL;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        //cond is only part of while loops
        if(nodeDef.getName().contains("/cond/"))
            return;
        //usually should be a merge node for a conditional
        val ifNodes = TFGraphMapper.getInstance().nodesForIf(nodeDef,graph);


        val trueScopeGraphDefBuilder = GraphDef.newBuilder();
        for(val node : ifNodes.getTrueNodes())  {
            trueScopeGraphDefBuilder.addNode(node);
        }


        val trueScope = TFGraphMapper.getInstance().importGraph(trueScopeGraphDefBuilder.build());


        val falseScopeGraphDefBuilder = GraphDef.newBuilder();
        for(val node : ifNodes.getFalseNodes())  {
            falseScopeGraphDefBuilder.addNode(node);

        }

        val falseScope = TFGraphMapper.getInstance().importGraph(falseScopeGraphDefBuilder.build());


        val condScopeGraphDefBuilder = GraphDef.newBuilder();
        for(val node : ifNodes.getCondNodes())  {
            condScopeGraphDefBuilder.addNode(node);

        }


        val condScope = TFGraphMapper.getInstance().importGraph(condScopeGraphDefBuilder.build());



        initWith.putSubFunction(ifNodes.getTrueBodyScopeName(),trueScope);
        initWith.putSubFunction(ifNodes.getFalseBodyScopeName(),falseScope);
        initWith.putSubFunction(ifNodes.getConditionBodyScopeName(),condScope);

        this.loopBodyExecution = trueScope;
        this.falseBodyExecution = falseScope;
        this.predicateExecution = condScope;
    }


    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {

    }



    @Override
    public List<long[]> calculateOutputShape() {
        return Arrays.asList(new long[]{1,1});
    }

    @Override
    public CustomOpDescriptor getDescriptor() {
        return null;
    }

    @Override
    public void assertValidForExecution() {

    }

    @Override
    public void populateInputsAndOutputsFromSameDiff() {

    }



    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("This operation has no TF counterpart");
    }
}
