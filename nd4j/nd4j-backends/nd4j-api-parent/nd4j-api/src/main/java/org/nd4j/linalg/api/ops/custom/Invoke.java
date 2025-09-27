/*
 * ******************************************************************************
 * *
 * *
 * * This program and the accompanying materials are made available under the
 * * terms of the Apache License, Version 2.0 which is available at
 * * https://www.apache.org/licenses/LICENSE-2.0.
 * *
 * * See the NOTICE file distributed with this work for additional
 * * information regarding copyright ownership.
 * * Unless required by applicable law or agreed to in writing, software
 * * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * * License for the specific language governing permissions and limitations
 * * under the License.
 * *
 * * SPDX-License-Identifier: Apache-2.0
 * *****************************************************************************
 */
package org.nd4j.linalg.api.ops.custom;

import lombok.Builder;
import lombok.Data;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.ExecutionResult;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

@Slf4j
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
        this.outputVarNames = invokeParams.outputVarNames;
        this.functionName = invokeParams.functionName;
        this.inputVarNames = invokeParams.inputVarNames;
        this.subGraphInputVarNames = invokeParams.subGraphInputVarNames;
        this.subGraphOutputVarNames = invokeParams.subGraphOutputVarNames;
        // Centralize initialization logic
        configureWithSameDiff(sameDiff);
    }

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
            Map<String,INDArray> inputMap = new LinkedHashMap<>();
            for(int i = 0; i < inputVarNameMappings.length; i++) {
                inputMap.put(subGraphInputNames[i],placeHolders.get(op.argNames()[i]));
            }

            Map<String, INDArray> output = instance.output(inputMap, relevantOutputNames);
            return ExecutionResult.builder()
                    .outputs(ExecutionResult.pack(output))
                    .build();
        } else {
            Map<String,SDValue> valueInputs = new LinkedHashMap<>();
            for(int i = 0; i < inputVarNameMappings.length; i++) {
                valueInputs.put(subGraphInputNames[i],valuePlaceHolders.get(op.argNames()[i]));
            }

            Map<String,SDValue> valueOutputs = instance.outputValues(valueInputs,relevantOutputNames);
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
            if (this.functionName == null) {
                throw new IllegalStateException("Invoke operation '" + this.getOwnName() +
                        "' has no function name set. Cannot determine output variables.");
            }

            SameDiff func = sameDiff.getFunction(this.functionName);
            if (func == null) {
                throw new IllegalArgumentException("Unable to determine output data types for variables. " +
                        "No function of '" + this.functionName + "' found in SameDiff instance!");
            }

            if (subGraphOutputVarNames == null) {
                // Try to auto-configure from the sub-graph's outputs if not explicitly set
                if (func.outputs() != null && !func.outputs().isEmpty()) {
                    log.warn("subGraphOutputVarNames is null for Invoke operation '{}' with function '{}'. " +
                                    "Auto-configuring from sub-graph outputs: {}",
                            this.getOwnName(), this.functionName, func.outputs());
                    this.subGraphOutputVarNames = func.outputs().toArray(new String[0]);
                } else {
                    throw new IllegalStateException("Invalid InvokeConfiguration found for operation '" +
                            this.getOwnName() + "'. Please specify sub graph output names. " +
                            "FunctionName: " + this.functionName + ", Sub-graph outputs: " +
                            (func.outputs() != null ? func.outputs() : "null"));
                }
            }

            SDVariable[] outputs = new SDVariable[subGraphOutputVarNames.length];
            for (int i = 0; i < subGraphOutputVarNames.length; i++) {
                String subGraphVarName = subGraphOutputVarNames[i];
                SDVariable variable = func.getVariable(subGraphVarName);
                if(variable == null) {
                    throw new IllegalStateException("No variable found in sub graph '" + this.functionName +
                            "' named '" + subGraphVarName + "' for Invoke operation '" + this.getOwnName() + "'");
                }

                String newVarName = outputVarNames != null && outputVarNames.length > i ?
                        outputVarNames[i] : (subGraphVarName + "_" + functionName);

                switch(variable.getVariableType()) {
                    case VARIABLE:
                    case ARRAY:
                    case PLACEHOLDER:
                    case SEQUENCE:
                    case CONSTANT:
                        if(variable.getShape() != null) {
                            outputs[i] = sameDiff.var(newVarName, variable.dataType(), variable.getShape());
                        } else {
                            outputs[i] = sameDiff.var(newVarName, variable.dataType());
                        }
                        outputs[i].setVariableType(VariableType.ARRAY);
                        break;
                }
            }
            this.outputVariables = outputs;
        }
        return outputVariables;
    }

    @Override
    public int getNumOutputs() {
        if(subGraphOutputVarNames != null)
            return subGraphOutputVarNames.length;
        else if(outputVarNames != null)
            return outputVarNames.length;

        if (sameDiff != null && functionName != null) {
            SameDiff func = sameDiff.getFunction(functionName);
            if(func != null && func.outputs() != null) {
                return func.outputs().size();
            }
        }
        return 1;
    }

    @Override
    public String opName() {
        return "invoke";
    }

    /**
     * This is the central point for finalizing the op's configuration.
     * It's called by the constructor for programmatic creation and by the
     * deserialization process after properties have been loaded from the file.
     */
    @Override
    public void configureWithSameDiff(SameDiff sameDiff) {
        // 1. Link the op to the graph instance. THIS MUST BE FIRST.
        super.configureWithSameDiff(sameDiff);

        // 2. Skip output variable calculation if we don't have the required configuration yet
        if (this.subGraphOutputVarNames == null || this.functionName == null) {
            log.debug("Invoke operation '{}' configuration incomplete during configureWithSameDiff. " +
                            "FunctionName: {}, SubGraphOutputVarNames: {}. Skipping output variable calculation.",
                    this.getOwnName(), this.functionName,
                    this.subGraphOutputVarNames != null ? Arrays.toString(this.subGraphOutputVarNames) : "null");
            return; // Skip output variable calculation for now
        }

        // 3. If outputs haven't been calculated yet and we have the required config, do it now.
        if (this.outputVariables == null) {
            try {
                // Calculate and cache the output variables
                outputVariables();
                // Register the newly created variables with the graph
                addOutputsToOp();
            } catch (Exception e) {
                log.warn("Failed to configure output variables for Invoke operation '{}': {}",
                        this.getOwnName(), e.getMessage());
            }
        }
    }


    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        if (properties == null || properties.isEmpty()) {
            log.warn("Empty or null properties map for Invoke operation '{}'", this.getOwnName());
            return;
        }

        log.info("INVOKE PROPERTIES DEBUG: Setting properties for '{}': {}", this.getOwnName(), properties);

        for (Map.Entry<String, Object> entry : properties.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();

            log.info("INVOKE PROPERTIES DEBUG: Processing property '{}' = {} (type: {})",
                    key, value, value != null ? value.getClass().getSimpleName() : "null");

            if (value == null) continue;

            switch (key) {
                case "functionName":
                    if(value instanceof String[]) {
                        String[] arr = (String[])  value;
                        this.functionName = arr[0];
                    } else {
                        this.functionName = (String) value;

                    }
                    log.info("INVOKE PROPERTIES DEBUG: Set functionName = '{}'", this.functionName);
                    break;
                case "subGraphOutputVarNames":
                    if (value instanceof String[]) {
                        this.subGraphOutputVarNames = (String[]) value;
                        log.info("INVOKE PROPERTIES DEBUG: Set subGraphOutputVarNames = {}", Arrays.toString(this.subGraphOutputVarNames));
                    } else if (value instanceof String) {
                        this.subGraphOutputVarNames = new String[]{(String) value};
                        log.info("INVOKE PROPERTIES DEBUG: Set subGraphOutputVarNames from single string = {}", Arrays.toString(this.subGraphOutputVarNames));
                    }
                    break;
                case "inputVarNames":
                    if (value instanceof String[]) {
                        this.inputVarNames = (String[]) value;
                    } else if (value instanceof String) {
                        this.inputVarNames = new String[]{(String) value};
                    }
                    break;
                case "outputVarNames":
                    if (value instanceof String[]) {
                        this.outputVarNames = (String[]) value;
                    } else if (value instanceof String) {
                        this.outputVarNames = new String[]{(String) value};
                    }
                    break;
                case "subGraphInputVarNames":
                    if (value instanceof String[]) {
                        this.subGraphInputVarNames = (String[]) value;
                    } else if (value instanceof String) {
                        this.subGraphInputVarNames = new String[]{(String) value};
                    }
                    break;
                default:
                    log.warn("Unknown property '{}' for Invoke operation '{}'", key, this.getOwnName());
                    break;
            }
        }

        log.info("INVOKE PROPERTIES DEBUG: Final state - functionName: '{}', subGraphOutputVarNames: {}",
                this.functionName, this.subGraphOutputVarNames != null ? Arrays.toString(this.subGraphOutputVarNames) : "null");
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String, Object> ret = new LinkedHashMap<>();
        ret.put("functionName", functionName);
        ret.put("inputVarNames", inputVarNames);
        ret.put("outputVarNames", outputVarNames);
        ret.put("subGraphInputVarNames", subGraphInputVarNames);
        ret.put("subGraphOutputVarNames", subGraphOutputVarNames);
        return ret;
    }



    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        List<DataType> ret = new ArrayList<>();
        if (sameDiff != null && functionName != null) {
            SameDiff func = sameDiff.getFunction(functionName);
            if (func != null && subGraphOutputVarNames != null) {
                for (String varName : subGraphOutputVarNames) {
                    SDVariable v = func.getVariable(varName);
                    if (v != null) {
                        ret.add(v.dataType());
                    } else {
                        ret.add(DataType.FLOAT);
                    }
                }
                return ret;
            }
        }

        for(int i = 0; i < getNumOutputs(); i++)
            ret.add(DataType.FLOAT);
        return ret;
    }

    @Override
    public List<DataBuffer> calculateOutputShape(OpContext oc) {
        List<LongShapeDescriptor> shapeDescriptors = new ArrayList<>();
        if(sameDiff != null && functionName != null) {
            SameDiff func = sameDiff.getFunction(functionName);
            if (func != null && subGraphOutputVarNames != null) {
                for (String varName : subGraphOutputVarNames) {
                    SDVariable v = func.getVariable(varName);
                    if (v != null && v.getShape() != null) {
                        shapeDescriptors.add(LongShapeDescriptor.fromShape(v.getShape(), v.dataType()));
                    } else {
                        shapeDescriptors.add(LongShapeDescriptor.fromShape(new long[]{1}, DataType.FLOAT));
                    }
                }
            }
        }

        if (shapeDescriptors.isEmpty()) {
            for(int i = 0; i < getNumOutputs(); i++) {
                shapeDescriptors.add(LongShapeDescriptor.fromShape(new long[]{1}, DataType.FLOAT));
            }
        }

        List<DataBuffer> outputBuffers = new ArrayList<>();
        for (LongShapeDescriptor descriptor : shapeDescriptors) {
            long[] shapeInfo = descriptor.toShapeInfo();
            outputBuffers.add(Nd4j.createBuffer(shapeInfo));
        }

        return outputBuffers;
    }
}