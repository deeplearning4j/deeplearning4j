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
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.CustomOpDescriptor;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalArgumentException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.impl.ZeroInitScheme;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Equivalent to tensorflow's while loop
 * Takes in:
 * loopVars
 * loop body
 * condition
 *
 * runs loop till condition is false.
 * @author Adam Gibson
 */
@NoArgsConstructor
@Slf4j
public class While extends DifferentialFunction implements CustomOp {
    private AtomicInteger  startPosition;



    @Getter
    protected SameDiff loopBodyExecution,predicateExecution;


    @Getter
    protected SameDiff.SameDiffConditional predicate;
    @Getter
    protected SameDiff.SameDiffFunctionDefinition trueBody;

    @Getter
    protected String blockName,trueBodyName;

    @Getter
    protected SDVariable[] inputVars;


    @Getter
    protected SDVariable targetBoolean;

    protected SDVariable dummyResult;

    @Getter
    @Setter
    protected SDVariable[] outputVars;

    @Getter
    protected int numLooped = 0;

    /**
     * Mainly meant for tensorflow import.
     * This allows {@link #initFromTensorFlow(NodeDef, SameDiff, Map, GraphDef)}
     * to continue from a parent while loop
     * using the same graph
     * @param startPosition the start position for the import scan
     */
    public While(AtomicInteger startPosition) {
        this.startPosition = startPosition;
    }

    public While(While whileStatement) {
        this.sameDiff = whileStatement.sameDiff;
        this.outputVars = whileStatement.outputVars;
        this.loopBodyExecution = whileStatement.loopBodyExecution;
        this.numLooped = whileStatement.numLooped;
        this.dummyResult = whileStatement.dummyResult;
        this.predicate = whileStatement.predicate;
        this.predicateExecution = whileStatement.predicateExecution;
        this.inputVars = whileStatement.inputVars;
        this.dummyResult =  this.sameDiff.var("dummyresult-" + UUID.randomUUID().toString(),new long[]{1,1},new ZeroInitScheme('f'));


    }



    @Builder
    public While(String blockName,
                 SameDiff parent,
                 SDVariable[] inputVars,
                 SameDiff.SameDiffConditional predicate,
                 SameDiff.SameDiffFunctionDefinition condition,
                 SameDiff.SameDiffFunctionDefinition trueBody) {
        init(blockName,parent,inputVars,predicate,condition,trueBody);
    }


    private void init(String blockName,
                      SameDiff parent,
                      SDVariable[] inputVars,
                      SameDiff.SameDiffConditional predicate,
                      SameDiff.SameDiffFunctionDefinition condition,
                      SameDiff.SameDiffFunctionDefinition trueBody) {
        this.sameDiff = parent;
        this.inputVars = inputVars;
        this.predicate = predicate;
        this.trueBody = trueBody;
        this.blockName = blockName;
        this.dummyResult =  parent.var("dummyresult-" + UUID.randomUUID().toString(),new long[]{1,1},new ZeroInitScheme('f'));
        parent.putFunctionForId(getOwnName(),this);

        parent.addArgsFor(inputVars,this);
        parent.addOutgoingFor(new SDVariable[]{dummyResult},this);


        //create a samediff sub graph for running just the execution
        //return a reference to the loop for referencing during actual execution
        SameDiff sameDiff = SameDiff.create();
        //store the reference to the result array and the same diff execution instance
        this.targetBoolean = predicate.eval(sameDiff,condition, inputVars);
        this.predicateExecution = sameDiff;
        //store references to the loop body
        String trueBodyName = "true-body-" + UUID.randomUUID().toString();
        this.trueBodyName = trueBodyName;
        //running define function will setup a proper same diff instance
        parent.defineFunction(trueBodyName,trueBody,inputVars);
        parent.defineFunction(blockName,condition,inputVars);
        parent.putSubFunction("predicate-eval-body",sameDiff);
        //get a reference to the actual loop body
        this.loopBodyExecution = parent.getFunction(trueBodyName);

    }


    @Override
    public SDVariable[] outputVariables(String baseName) {
        return new SDVariable[]{dummyResult};
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> ret = new ArrayList<>();
        ret.addAll(Arrays.asList(new WhileDerivative(this).outputVariables()));
        return ret;
    }



    /**
     * Increments the loop counter.
     * This should be called when the loop
     * actually executes.
     */
    public void incrementLoopCounter() {
        numLooped++;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        doImport(nodeDef,initWith,attributesForNode,graph,new LinkedHashSet<String>(),new AtomicInteger(0));
    }


    private  void doImport(NodeDef nodeDef,SameDiff initWith,Map<String,AttrValue> attributesForNode,GraphDef graph,Set<String> skipSet,AtomicInteger currIndex) {
        val uniqueId = java.util.UUID.randomUUID().toString();
        skipSet.add(nodeDef.getName());
        val scopeCondition = SameDiff.create();
        val scopeLoop = SameDiff.create();
        initWith.putSubFunction("condition-" + uniqueId,scopeCondition);
        initWith.putSubFunction("loopbody-" + uniqueId,scopeLoop);
        this.loopBodyExecution = scopeLoop;
        this.predicateExecution = scopeCondition;
        this.startPosition = currIndex;

        log.info("Adding 2 new scopes for WHILE {}");


        val nodes = graph.getNodeList();

        /**
         * Plan is simple:
         * 1) we read all declarations of variables used within loop
         * 2) we set up conditional scope
         * 3) we set up body scope
         * 4) ???
         * 5) PROFIT!
         */

        for (; currIndex.get() < nodes.size(); currIndex.incrementAndGet()) {
            val tfNode = nodes.get(currIndex.get());

            if (!tfNode.getOp().equalsIgnoreCase("enter")) {
                //skipSet.add(tfNode.getName());
                break;
            }

//            if (skipSet.contains(tfNode.getName()))
//                continue;

            skipSet.add(tfNode.getName());

            val vars = new SDVariable[tfNode.getInputCount()];
            for (int e = 0; e < tfNode.getInputCount(); e++) {
                val input = TFGraphMapper.getInstance().getNodeName(tfNode.getInput(e));
                vars[e] = initWith.getVariable(input) == null ? initWith.var(input,null,new ZeroInitScheme()) : initWith.getVariable(input);
                scopeCondition.var(vars[e]);
                scopeLoop.var(vars[e]);
            }

            this.inputVars = vars;
        }


        // now we're skipping Merge step, since we've already captured variables at Enter step
        int mergedCnt = 0;
        for (; currIndex.get() < nodes.size(); currIndex.incrementAndGet()) {
            val tfNode = nodes.get(currIndex.get());

            if (!tfNode.getOp().equalsIgnoreCase("merge")) {
                scopeLoop.var(TFGraphMapper.getInstance().getNodeName(tfNode.getName()),null,new ZeroInitScheme());
                break;
            }

            skipSet.add(tfNode.getName());
            val var = scopeLoop.var(TFGraphMapper.getInstance().getNodeName(tfNode.getName()),null,new ZeroInitScheme());
            scopeCondition.var(var);
            initWith.var(var);
            mergedCnt++;
        }


        // now, we're adding conditional scope
        for (; currIndex.get() < nodes.size(); currIndex.incrementAndGet()) {
            val tfNode = nodes.get(currIndex.get());

            // we're parsing up to condition
            if (tfNode.getOp().equalsIgnoreCase("LoopCond")) {
                skipSet.add(tfNode.getName());
                currIndex.incrementAndGet();
                break;
            }

            boolean isConst = tfNode.getOp().equalsIgnoreCase("const");
            boolean isVar = tfNode.getOp().startsWith("VariableV");
            boolean isPlaceholder = tfNode.getOp().startsWith("Placeholder");


            if (isConst || isVar || isPlaceholder) {
                val var = scopeCondition.var(tfNode.getName(),null,new ZeroInitScheme());
                scopeLoop.var(var);
                initWith.var(var);
                log.info("Adding condition var [{}]", var.getVarName());

            }
            else if(!skipSet.contains(tfNode.getName())) {
                val func = DifferentialFunctionClassHolder.getInstance().getInstance(TFGraphMapper.getInstance().getMappedOp(tfNode.getOp()).opName());
                func.initFromTensorFlow(tfNode,scopeCondition,nodeDef.getAttrMap(),graph);
                func.setSameDiff(scopeLoop);

            }

            skipSet.add(tfNode.getName());
        }



        // time to skip some Switch calls
        int switchCnt = 0;
        for (; currIndex.get() < nodes.size(); currIndex.incrementAndGet()) {
            val tfNode = nodes.get(currIndex.get());

            // we're parsing up to condition
            if (!tfNode.getOp().equalsIgnoreCase("Switch"))
                break;

            switchCnt++;
            skipSet.add(tfNode.getName());
        }

        // now we're parsing Identity step
        int identityCnt = 0;
        for (; currIndex.get() < nodes.size(); currIndex.incrementAndGet()) {
            val tfNode = nodes.get(currIndex.get());


            if (!tfNode.getOp().equalsIgnoreCase("Identity")) {
                break;
            }


            val func = DifferentialFunctionClassHolder.getInstance().getInstance(TFGraphMapper.getInstance().getMappedOp(tfNode.getOp()).opName());
            func.initFromTensorFlow(tfNode,initWith,nodeDef.getAttrMap(),graph);
            func.setSameDiff(scopeLoop);


            val variables = new SDVariable[tfNode.getInputCount()];
            for(int i = 0; i < tfNode.getInputCount(); i++) {
                val testVar = initWith.getVariable(TFGraphMapper.getInstance().getNodeName(tfNode.getInput(i)));
                if(testVar == null) {
                    variables[i] = initWith.var(tfNode.getInput(i),null,new ZeroInitScheme());
                    scopeCondition.var(variables[i]);
                    scopeLoop.var(variables[i]);
                    continue;
                }
                else {

                    variables[i] = initWith.getVariable(TFGraphMapper.getInstance().getNodeName(tfNode.getInput(i)));
                    scopeCondition.var(variables[i]);
                    scopeLoop.var(variables[i]);
                }

            }

            scopeLoop.addArgsFor(variables,func);
            skipSet.add(tfNode.getName());
        }


        // parsing body scope
        for (; currIndex.get() < nodes.size(); currIndex.incrementAndGet()) {
            val tfNode = nodes.get(currIndex.get());

            if (skipSet.contains(tfNode.getName())) {
                log.info("Skipping: {}", tfNode.getName());
                continue;
            }

            if (tfNode.getOp().equalsIgnoreCase("NextIteration")) {
//                skipSet.add(tfNode.getName());
                break;
            }

            if (skipSet.contains(tfNode.getName())) {
                log.info("Skipping: {}", tfNode.getName());
                continue;
            }



            boolean isConst = tfNode.getOp().equalsIgnoreCase("const");
            boolean isVar = tfNode.getOp().startsWith("VariableV");
            boolean isPlaceholder = tfNode.getOp().startsWith("Placeholder");


            if (isConst || isVar || isPlaceholder) {
                val var = scopeLoop.var(tfNode.getName(), null,new ZeroInitScheme());
                log.info("Adding body var [{}]",var.getVarName());

            } else {
                log.info("starting on [{}]: {}", tfNode.getName(), tfNode.getOp());

                if (tfNode.getOp().equalsIgnoreCase("enter")) {
                    log.info("NEW LOOP ----------------------------------------");
                    val func = new While(currIndex);
                    func.doImport(nodeDef,initWith,attributesForNode,graph,skipSet,currIndex);
                    func.setSameDiff(initWith);
                    log.info("END LOOP ----------------------------------------");
                } else {
                    val func = DifferentialFunctionClassHolder.getInstance().getInstance(TFGraphMapper.getInstance().getMappedOp(tfNode.getOp()).opName());

                    func.initFromTensorFlow(tfNode,initWith,nodeDef.getAttrMap(),graph);


                    func.setSameDiff(scopeCondition);

                    val variables = new SDVariable[tfNode.getInputCount()];
                    for(int i = 0; i < tfNode.getInputCount(); i++) {
                        val name = TFGraphMapper.getInstance().getNodeName(tfNode.getInput(i));
                        variables[i] = scopeCondition.getVariable(name);
                        if(variables[i] == null) {
                            if(scopeLoop.getVariable(name) == null)
                                variables[i] = scopeCondition.var(initWith.getVariable(name));
                            else if(scopeLoop.getVariable(name) != null)
                                variables[i] = scopeLoop.getVariable(name);
                            else
                                variables[i] = scopeLoop.var(name, Nd4j.scalar(1.0));
                        }
                    }

                    scopeLoop.addArgsFor(variables,func);


                }
            }

            skipSet.add(tfNode.getName());
        }


        val returnInputs = new ArrayList<SDVariable>();
        val returnOutputs = new ArrayList<SDVariable>();

        // mapping NextIterations, to Return op
        for (; currIndex.get() < nodes.size(); currIndex.incrementAndGet()) {
            val tfNode = nodes.get(currIndex.get());

            if (!tfNode.getOp().equalsIgnoreCase("NextIteration"))
                break;

            skipSet.add(tfNode.getName());

            val inputName = TFGraphMapper.getInstance().getNodeName(tfNode.getName());
            val input = initWith.getVariable(inputName) == null ? initWith.var(inputName,null,new ZeroInitScheme()) : initWith.getVariable(inputName) ;
            returnInputs.add(input);
        }


        this.outputVars = returnOutputs.toArray(new SDVariable[returnOutputs.size()]);
        this.inputVars = returnInputs.toArray(new SDVariable[returnInputs.size()]);
        initWith.addArgsFor(inputVars,this);
        initWith.addOutgoingFor(outputVars,this);

        // we should also map While/Exit to libnd4j while
        int exitCnt = 0;
        for (; currIndex.get() < nodes.size(); currIndex.incrementAndGet()) {
            val tfNode = nodes.get(currIndex.get());

            if (!tfNode.getOp().equalsIgnoreCase("Exit")) {
                //skipSet.add(tfNode.getName());
                break;
            }

            skipSet.add(tfNode.getName());
            val inputName = TFGraphMapper.getInstance().getNodeName(tfNode.getName());
            val input = initWith.getVariable(inputName) == null ? initWith.var(inputName,null,new ZeroInitScheme()) : initWith.getVariable(inputName) ;
        }


        //the output of the condition should always be a singular scalar
        //this is a safe assumption
        val conditionVars = scopeCondition.functions();
        if(conditionVars.length < 1) {
            throw new ND4JIllegalArgumentException("No functions found!");
        }
        this.targetBoolean = conditionVars[conditionVars.length - 1].outputVariables()[0];

        log.info("-------------------------------------------");

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {

    }


    @Override
    public String toString() {
        return opName();
    }

    @Override
    public String opName() {
        return "while";
    }

    @Override
    public long opHash() {
        return opName().hashCode();
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
    public List<long[]> calculateOutputShape() {
        List<long[]> ret =  new ArrayList<>();
        for(SDVariable var : args()) {
            ret.add(sameDiff.getShapeForVarName(var.getVarName()));
        }
        return ret;
    }

    @Override
    public CustomOpDescriptor getDescriptor() {
        return CustomOpDescriptor.builder().build();
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
        throw new NoOpNameFoundException("No *singular (eg: use tensorflowNames() found for this op " + opName());
    }

    @Override
    public String[] tensorflowNames() {
        throw new NoOpNameFoundException("This operation has no TF counterpart");
    }


    @Override
    public Op.Type opType() {
        return Op.Type.LOOP;
    }
}
