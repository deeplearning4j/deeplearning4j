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
package org.nd4j.linalg.api.ops.custom;

import lombok.Builder;
import lombok.Data;
import lombok.Getter;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.ExecutionResult;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.autodiff.samediff.config.SDValueType;
import org.nd4j.autodiff.samediff.internal.AbstractSession;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Invoke is an op
 */
public class Invoke extends DynamicCustomOp {

    @Getter
    private String functionName;
    @Getter
    private String[] inputVarNames;
    @Getter
    private String[] outputVarNames;
    @Getter
    private String[] subGraphInputVarNames;
    @Getter
    private String[] subGraphOutputVarNames;

    public Invoke() {
    }

    @Data
    @Builder
    public static class InvokeParams {
        private String functionName;
        private SDVariable[] inputs;
        private String[] inputVarNames;
        private String[] outputVarNames;
        private String[] subGraphInputVarNames;
        private String[] subGraphOutputVarNames;
    }


    public Invoke(SameDiff sameDiff,InvokeParams invokeParams) {
        super(sameDiff,invokeParams.inputs);
        this.sameDiff = sameDiff;
        this.outputVarNames = invokeParams.outputVarNames;
        this.functionName = invokeParams.functionName;
        this.inputVarNames = invokeParams.inputVarNames;
        this.subGraphInputVarNames = invokeParams.subGraphInputVarNames;
        this.subGraphOutputVarNames = invokeParams.subGraphOutputVarNames;
    }

    /**
     * Perform the invoke method.
     * @param op the {@link Invoke} instance to use
     * @param placeHolders the singular placeholders to pass in to the function
     * @param valuePlaceHolders the value placeholders to pass in to the function
     * @return the {@link ExecutionResult} from the sub function
     */
    public static ExecutionResult doInvoke(DifferentialFunction op, Map<String,INDArray> placeHolders, Map<String, SDValue> valuePlaceHolders) {
        Invoke invoke = (Invoke) op;
        String funcName = invoke.getFunctionName();
        SameDiff instance = op.getSameDiff().getFunction(funcName);
        SDVariable[] args = op.args();
        String[] inputVarNameMappings = invoke.getInputVarNames();

        String[] subGraphInputNames = invoke.subGraphInputVarNames;
        if(subGraphInputNames == null)
            subGraphInputNames = inputVarNameMappings;

        SDVariable[] outputs = op.outputVariables();

        if(inputVarNameMappings == null) {
            inputVarNameMappings = new String[args.length];
            //default to input names of op unless specified
            for(int i = 0; i < inputVarNameMappings.length; i++) {
                inputVarNameMappings[i] = args[i].name();
            }
        }

        String[] outputVarNameMappings = invoke.getOutputVarNames();
        if(outputVarNameMappings == null) {
            outputVarNameMappings = new String[outputs.length];
            for(int i = 0; i < outputs.length; i++) {
                outputVarNameMappings[i] = outputs[i].name();
            }
        }


        String[] subGraphOutputNames = invoke.subGraphOutputVarNames;
        if(subGraphOutputNames == null)
            subGraphOutputNames = outputVarNameMappings;



        List<String> relevantOutputNames = Arrays.asList(subGraphOutputNames);
        if(valuePlaceHolders.isEmpty()) {
            INDArray[] retOutput = new INDArray[subGraphOutputNames.length];
            Map<String,INDArray> inputMap = new LinkedHashMap<>();
            for(int i = 0; i < inputVarNameMappings.length; i++) {
                //note that we use the inputs in numerical order ignoring the names
                //this is because the input names aren't aligned with what's passed in
                inputMap.put(subGraphInputNames[i],placeHolders.get(op.argNames()[i]));
            }

            Map<String, INDArray> output = instance.output(inputMap, relevantOutputNames);
            //note not all keys maybe the same as what we expect so we only add the keys we care about
            int numAdded = 0;
            for(Map.Entry<String,INDArray> result : output.entrySet()) {
                if(relevantOutputNames.contains(result.getKey())) {
                    retOutput[numAdded] = output.get(result.getKey());
                    numAdded++;
                }
            }

            return ExecutionResult.builder()
                    .outputs(ExecutionResult.pack(output))
                    .build();
        } else {
            Map<String,SDValue> valueInputs = new LinkedHashMap<>();
            for(int i = 0; i < inputVarNameMappings.length; i++) {
                //note that we use the inputs in numerical order ignoring the names
                //this is because the input names aren't aligned with what's passed in
                valueInputs.put(subGraphInputNames[i],valuePlaceHolders.get(op.argNames()[i]));
            }

            Map<String,SDValue> valueOutputs = instance.outputValues(valueInputs,relevantOutputNames);
            //rearrange to be in right order for return, this is critical
            Map<String,SDValue> result = new LinkedHashMap<>();
            for(int i = 0; i < outputVarNameMappings.length; i++) {
                result.put(outputs[i].name(), valueOutputs.get(subGraphOutputNames[i]));
            }

            return ExecutionResult.builder()
                    .valueOutputs(result)
                    .build();

        }

    }

    @Override
    public SDVariable[] outputVariables() {
        if(outputVariables == null) {
            SameDiff func = sameDiff.getFunction(this.functionName);
            if (func == null) {
                throw new IllegalArgumentException("Unable to determine output data types for variables. No function of " + this.functionName + " found!");
            }

            if (subGraphOutputVarNames == null) {
                throw new IllegalStateException("Invalid InvokeConfiguration found. Please specify sub graph output names.");
            }

            SDVariable[] outputs = new SDVariable[subGraphOutputVarNames.length];
            for (int i = 0; i < subGraphOutputVarNames.length; i++) {
                String subGraphVarName = subGraphOutputVarNames[i];
                SDVariable variable = func.getVariable(subGraphVarName);
                if(variable == null) {
                    throw new IllegalStateException("No variable found in sub graph named " + subGraphVarName);
                }
                switch(variable.getVariableType()) {
                    case VARIABLE:
                    case ARRAY:
                    case PLACEHOLDER:
                    case SEQUENCE:
                        if(variable.getShape() != null) {
                            SDVariable clone2 = sameDiff.var(subGraphVarName + "_" + functionName, variable.dataType(), variable.getShape());
                            clone2.setVariableType(VariableType.ARRAY);
                            outputs[i] = clone2;
                        } else { //placeholder shape
                            SDVariable clone2 = sameDiff.var(subGraphVarName + "_" + functionName, variable.dataType());
                            clone2.setVariableType(VariableType.ARRAY);
                            outputs[i] = clone2;
                        }
                        break;
                    case CONSTANT:
                        SDVariable clone2 = sameDiff.var(subGraphVarName + "_" + functionName, variable.dataType());
                        clone2.setVariableType(VariableType.ARRAY);
                        outputs[i] = clone2;
                        break;

                }

            }

            this.outputVariables = outputs;

            if (outputVarNames != null && outputVarNames.length == outputs.length)
                for (int i = 0; i < outputs.length; i++) {
                    if (!outputs[i].name().equals(outputVarNames[i])) {
                        sameDiff.updateVariableNameAndReference(outputs[i], outputVarNames[i], true);
                    }
                }
            else if (this.outputVariables == null) {
                throw new IllegalArgumentException("Invalid configuration for output variable names. Must be equal to the number of outputs.");
            }

            //add outgoing ops after generating output variables
            addOutputsToOp();

            return outputs;
        }
        return outputVariables;
    }

    @Override
    public int getNumOutputs() {
        if(subGraphOutputVarNames != null)
            return subGraphOutputVarNames.length;
        else if(outputVarNames != null)
            return outputVarNames.length;
        return 1;
    }

    @Override
    public String opName() {
        return "invoke";
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
    }

    @Override
    public void configureWithSameDiff(SameDiff sameDiff) {
        super.configureWithSameDiff(sameDiff);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        List<DataType> ret = new ArrayList<>();
        for(int i = 0; i < getNumOutputs(); i++)
            ret.add(DataType.FLOAT);
        return ret;
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        return Collections.emptyList();
    }


    @Override
    public List<LongShapeDescriptor> calculateOutputShape(OpContext oc) {
        /**
         * TODO: Figure out how to invoke calculate output shape
         * for a graph. This may involve adding a new function
         * to a samediff graph that just calls compute shape for everything.
         */
        List<LongShapeDescriptor> ret = new ArrayList<>();
        for(int i = 0; i < getNumOutputs(); i++) {
            ret.add(LongShapeDescriptor.fromShape(new int[]{1},DataType.DOUBLE));
        }


        return ret;
    }
}
