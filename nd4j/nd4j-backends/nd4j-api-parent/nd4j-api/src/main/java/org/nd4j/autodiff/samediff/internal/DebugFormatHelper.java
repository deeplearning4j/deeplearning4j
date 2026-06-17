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

package org.nd4j.autodiff.samediff.internal;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.autodiff.samediff.config.SDValueType;
import org.nd4j.autodiff.samediff.execution.ExecutionNode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

public class DebugFormatHelper {

    /**
     * Formats a shape array as a readable string for debug output.
     */
    public static String formatShape(long[] shape) {
        if (shape == null) return "null";
        if (shape.length == 0) return "[]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(shape[i]);
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * Formats input information for debug output.
     */
    public static String formatInputsDebug(ExecutionNode node, Map<String, SDValue> variableValues) {
        StringBuilder sb = new StringBuilder();
        List<String> inputs = node.getInputVariables();
        if (inputs == null || inputs.isEmpty()) {
            sb.append("    Inputs: (none)\n");
        } else {
            sb.append("    Inputs:\n");
            for (int i = 0; i < inputs.size(); i++) {
                String inputName = inputs.get(i);
                SDValue value = variableValues.get(inputName);
                sb.append("      [").append(i).append("] ").append(inputName).append(": ");
                if (value == null) {
                    sb.append("null");
                } else if (value.getSdValueType() == SDValueType.TENSOR) {
                    INDArray arr = value.getTensorValue();
                    if (arr == null) {
                        sb.append("null tensor");
                    } else {
                        sb.append("shape=").append(formatShape(arr.shape()))
                          .append(", dtype=").append(arr.dataType());
                    }
                } else if (value.getSdValueType() == SDValueType.LIST) {
                    List<INDArray> list = value.getListValue();
                    sb.append("list[").append(list != null ? list.size() : 0).append("]");
                } else {
                    sb.append(value.getSdValueType());
                }
                sb.append("\n");
            }
        }
        return sb.toString();
    }

    /**
     * Formats output information for debug output.
     */
    public static String formatOutputsDebug(ExecutionNode node, Map<String, SDValue> variableValues,
                                            SameDiff sameDiff) {
        StringBuilder sb = new StringBuilder();
        List<String> outputs = node.getOutputVariables();
        if (outputs == null || outputs.isEmpty()) {
            sb.append("    Outputs: (none)\n");
        } else {
            sb.append("    Outputs:\n");
            for (int i = 0; i < outputs.size(); i++) {
                String outputName = outputs.get(i);
                SDVariable outVar = sameDiff.getVariable(outputName);
                sb.append("      [").append(i).append("] ").append(outputName);
                if (outVar != null) {
                    sb.append(": dtype=").append(outVar.dataType());
                    if (outVar.getShape() != null) {
                        sb.append(", expectedShape=").append(formatShape(outVar.getShape()));
                    }
                }
                // Check if already computed
                SDValue computed = variableValues.get(outputName);
                if (computed != null && computed.getSdValueType() == SDValueType.TENSOR) {
                    INDArray arr = computed.getTensorValue();
                    if (arr != null) {
                        sb.append(" -> actualShape=").append(formatShape(arr.shape()));
                    }
                }
                sb.append("\n");
            }
        }
        return sb.toString();
    }

    /**
     * Formats comprehensive node metadata for error messages.
     * This includes the operation name, input variable names with their shapes/values,
     * and expected output variable names.
     *
     * @param node the execution node that failed
     * @param variableValues current variable values map
     * @param sameDiff the SameDiff instance
     * @return formatted string with node metadata for error context
     */
    public static String formatNodeErrorContext(ExecutionNode node, Map<String, SDValue> variableValues,
                                                SameDiff sameDiff) {
        StringBuilder sb = new StringBuilder();
        sb.append("\n========================================\n");
        sb.append("SAMEDIFF NODE EXECUTION ERROR CONTEXT\n");
        sb.append("========================================\n");

        String opName = node.getOperationName();
        sb.append("Operation Name: ").append(opName).append("\n");

        // Get the SameDiff op for additional info
        SameDiffOp sameDiffOp = sameDiff.getOps().get(opName);
        if (sameDiffOp != null) {
            DifferentialFunction op = sameDiffOp.getOp();
            if (op != null) {
                sb.append("Op Type: ").append(op.opName()).append("\n");
                sb.append("Op Class: ").append(op.getClass().getSimpleName()).append("\n");
            }
        }

        // Input variables with details
        List<String> inputs = node.getInputVariables();
        sb.append("\nINPUT VARIABLES (").append(inputs != null ? inputs.size() : 0).append("):\n");
        if (inputs != null && !inputs.isEmpty()) {
            for (int i = 0; i < inputs.size(); i++) {
                String inputName = inputs.get(i);
                sb.append("  [").append(i).append("] '").append(inputName).append("'");

                SDValue value = variableValues.get(inputName);
                if (value == null) {
                    sb.append(" -> VALUE NOT FOUND IN CONTEXT");
                } else if (value.getSdValueType() == SDValueType.TENSOR) {
                    INDArray arr = value.getTensorValue();
                    if (arr == null) {
                        sb.append(" -> null tensor");
                    } else {
                        sb.append("\n      Shape: ").append(formatShape(arr.shape()));
                        sb.append(", DataType: ").append(arr.dataType());
                        sb.append(", Closed: ").append(arr.wasClosed());
                        if (arr.data() != null && !arr.wasClosed()) {
                            try {
                                sb.append("\n      Buffer Address: 0x").append(Long.toHexString(arr.data().pointer().address()));
                            } catch (Exception e) {
                                sb.append("\n      Buffer Address: <released>");
                            }
                        }
                        // For scalars, show the value
                        if (arr.isScalar() && !arr.wasClosed()) {
                            try {
                                if (arr.dataType() == DataType.INT64) {
                                    long val = arr.getLong(0);
                                    sb.append("\n      Scalar Value: ").append(val);
                                    sb.append(" (hex: 0x").append(Long.toHexString(val)).append(")");
                                } else if (arr.dataType().isFPType()) {
                                    sb.append("\n      Scalar Value: ").append(arr.getDouble(0));
                                }
                            } catch (Exception e) {
                                sb.append("\n      Scalar Value: <error reading value>");
                            }
                        }
                    }
                } else if (value.getSdValueType() == SDValueType.LIST) {
                    List<INDArray> list = value.getListValue();
                    sb.append(" -> list[").append(list != null ? list.size() : 0).append("]");
                } else {
                    sb.append(" -> ").append(value.getSdValueType());
                }
                sb.append("\n");
            }
        } else {
            sb.append("  (no inputs)\n");
        }

        // Output variables
        List<String> outputs = node.getOutputVariables();
        sb.append("\nOUTPUT VARIABLES (").append(outputs != null ? outputs.size() : 0).append("):\n");
        if (outputs != null && !outputs.isEmpty()) {
            for (int i = 0; i < outputs.size(); i++) {
                String outputName = outputs.get(i);
                sb.append("  [").append(i).append("] '").append(outputName).append("'");

                SDVariable outVar = sameDiff.getVariable(outputName);
                if (outVar != null) {
                    sb.append(" -> dtype=").append(outVar.dataType());
                    if (outVar.getShape() != null) {
                        sb.append(", expectedShape=").append(formatShape(outVar.getShape()));
                    }
                }
                sb.append("\n");
            }
        } else {
            sb.append("  (no outputs)\n");
        }

        sb.append("========================================\n");
        return sb.toString();
    }

    public static String formatValue(SDValue value) {
        if (value.getSdValueType() == SDValueType.TENSOR) {
            INDArray arr = value.getTensorValue();
            if (arr == null) return "null";
            if (arr.isScalar()) return String.valueOf(arr.getDouble(0));
            return Arrays.toString(arr.shape()) + " = " + arr.toString().replaceAll("\\s+", " ").trim();
        } else if (value.getSdValueType() == SDValueType.LIST) {
            return "List[" + value.getListValue().size() + "]";
        }
        return value.toString();
    }

    /**
     * Get all dependent values for a variable as a formatted string.
     *
     * @param variableValues Current variable values from execution
     * @param variableName Variable to get dependencies for
     * @param sameDiff the SameDiff instance
     * @return Formatted string with dependent values
     */
    public static String getDependentValuesString(Map<String, SDValue> variableValues, String variableName,
                                                   SameDiff sameDiff) {
        Map<String, String> deps = getDependentValuesMap(variableValues, variableName, sameDiff);
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, String> entry : deps.entrySet()) {
            sb.append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
        }
        return sb.toString();
    }

    /**
     * Get all dependent values for a variable as a map.
     *
     * @param variableValues Current variable values from execution
     * @param variableName Variable to get dependencies for
     * @param sameDiff the SameDiff instance
     * @return Map of variable names to their values
     */
    public static Map<String, String> getDependentValuesMap(Map<String, SDValue> variableValues,
                                                             String variableName, SameDiff sameDiff) {
        Map<String, String> result = new LinkedHashMap<>();
        Set<String> visited = new HashSet<>();
        collectDependentValues(variableValues, variableName, result, visited, sameDiff);
        return result;
    }

    public static void collectDependentValues(Map<String, SDValue> variableValues, String varName,
                                               Map<String, String> result, Set<String> visited,
                                               SameDiff sameDiff) {
        if (visited.contains(varName)) {
            return;
        }
        visited.add(varName);

        // Add this variable's value
        SDValue value = variableValues.get(varName);
        if (value != null) {
            result.put(varName, formatValue(value));
        }

        // Find the op that produces this variable
        for (SameDiffOp op : sameDiff.getOps().values()) {
            if (op.getOutputsOfOp() != null && op.getOutputsOfOp().contains(varName)) {
                // Collect values from all inputs
                if (op.getInputsToOp() != null) {
                    for (String input : op.getInputsToOp()) {
                        collectDependentValues(variableValues, input, result, visited, sameDiff);
                    }
                }
                break;
            }
        }
    }
}
