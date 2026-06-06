/*
 *  ******************************************************************************
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

package org.deeplearning4j.ui.module.samediff;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;

import java.util.*;

/**
 * Serializes a SameDiff graph into a JSON-compatible Map structure
 * for rendering with Cytoscape.js in the browser.
 */
public class SameDiffGraphSerializer {

    /**
     * Convert a SameDiff instance to a Cytoscape.js-compatible graph structure.
     *
     * @param sd the SameDiff instance
     * @return Map with keys: nodes, edges, inputs, outputs, modelType, opCount, varCount, paramCount
     */
    public static Map<String, Object> serialize(SameDiff sd) {
        List<Map<String, Object>> nodes = new ArrayList<>();
        List<Map<String, Object>> edges = new ArrayList<>();
        int edgeId = 0;

        // Enumerate ops as nodes
        for (Map.Entry<String, SameDiffOp> entry : sd.getOps().entrySet()) {
            SameDiffOp sdOp = entry.getValue();
            DifferentialFunction fn = sdOp.getOp();

            Map<String, Object> nodeData = new LinkedHashMap<>();
            nodeData.put("id", sdOp.getName());
            nodeData.put("label", fn.opName());
            nodeData.put("opType", fn.opName());
            nodeData.put("nodeType", "op");

            // Extract op properties
            Map<String, Object> props = new LinkedHashMap<>();
            try {
                Map<String, Object> fnProps = fn.propertiesForFunction();
                if (fnProps != null) {
                    for (Map.Entry<String, Object> pe : fnProps.entrySet()) {
                        props.put(pe.getKey(), String.valueOf(pe.getValue()));
                    }
                }
            } catch (Exception e) {
                // Some ops may not support propertiesForFunction
            }

            // For CustomOps, also extract iArgs/tArgs/bArgs
            if (fn instanceof CustomOp) {
                CustomOp cop = (CustomOp) fn;
                if (cop.iArgs() != null && cop.iArgs().length > 0) {
                    props.put("iArgs", Arrays.toString(cop.iArgs()));
                }
                if (cop.tArgs() != null && cop.tArgs().length > 0) {
                    props.put("tArgs", Arrays.toString(cop.tArgs()));
                }
                if (cop.bArgs() != null && cop.bArgs().length > 0) {
                    props.put("bArgs", Arrays.toString(cop.bArgs()));
                }
            }

            nodeData.put("properties", props);

            // Input/output names
            nodeData.put("inputs", sdOp.getInputsToOp() != null ? sdOp.getInputsToOp() : Collections.emptyList());
            nodeData.put("outputs", sdOp.getOutputsOfOp() != null ? sdOp.getOutputsOfOp() : Collections.emptyList());

            Map<String, Object> node = new LinkedHashMap<>();
            node.put("data", nodeData);
            node.put("classes", "op");
            nodes.add(node);
        }

        // Enumerate variables as nodes
        long totalParams = 0;
        for (SDVariable var : sd.variables()) {
            Map<String, Object> nodeData = new LinkedHashMap<>();
            String varId = "var-" + var.name();
            nodeData.put("id", varId);
            nodeData.put("label", var.name());
            nodeData.put("varName", var.name());
            nodeData.put("nodeType", "variable");

            VariableType vt = var.getVariableType();
            nodeData.put("varType", vt != null ? vt.name() : "UNKNOWN");

            // Shape
            long[] shape = null;
            try {
                shape = var.getShape();
            } catch (Exception e) {
                // Shape may not be available for ARRAY type pre-execution
            }
            if (shape != null) {
                nodeData.put("shape", Arrays.toString(shape));
                if (vt == VariableType.VARIABLE) {
                    long count = 1;
                    for (long s : shape) count *= s;
                    totalParams += count;
                }
            }

            // DataType
            try {
                nodeData.put("dataType", var.dataType() != null ? var.dataType().name() : "UNKNOWN");
            } catch (Exception e) {
                nodeData.put("dataType", "UNKNOWN");
            }

            // Constant value preview for small constants
            if (vt == VariableType.CONSTANT) {
                try {
                    INDArray arr = var.getArr();
                    if (arr != null && arr.length() <= 10) {
                        nodeData.put("value", arr.toStringFull());
                    } else if (arr != null) {
                        nodeData.put("value", "Constant [" + arr.length() + " elements]");
                    }
                } catch (Exception e) {
                    // ignore
                }
            }

            Map<String, Object> node = new LinkedHashMap<>();
            node.put("data", nodeData);

            // Assign CSS class based on variable type
            String cssClass = "variable";
            if (vt == VariableType.CONSTANT) cssClass = "constant";
            else if (vt == VariableType.PLACEHOLDER) cssClass = "placeholder";
            else if (vt == VariableType.ARRAY) cssClass = "array";
            node.put("classes", cssClass);
            nodes.add(node);

            // Build edges: variable -> consuming ops, producing op -> variable
            Variable meta = sd.getVariables().get(var.name());
            if (meta != null) {
                // If this variable is produced by an op, add edge: op -> variable
                String producerOp = meta.getOutputOfOp();
                if (producerOp != null) {
                    Map<String, Object> edgeData = new LinkedHashMap<>();
                    edgeData.put("id", "e" + (edgeId++));
                    edgeData.put("source", producerOp);
                    edgeData.put("target", varId);
                    Map<String, Object> edge = new LinkedHashMap<>();
                    edge.put("data", edgeData);
                    edges.add(edge);
                }

                // Variable -> consuming ops
                List<String> consumers = meta.getInputsForOp();
                if (consumers != null) {
                    for (String consumer : consumers) {
                        Map<String, Object> edgeData = new LinkedHashMap<>();
                        edgeData.put("id", "e" + (edgeId++));
                        edgeData.put("source", varId);
                        edgeData.put("target", consumer);
                        Map<String, Object> edge = new LinkedHashMap<>();
                        edge.put("data", edgeData);
                        edges.add(edge);
                    }
                }
            }
        }

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("nodes", nodes);
        result.put("edges", edges);
        result.put("inputs", sd.inputs());
        result.put("outputs", sd.outputs());
        result.put("modelType", "SameDiff");
        result.put("opCount", sd.getOps().size());
        result.put("varCount", sd.variables().size());
        result.put("paramCount", totalParams);

        return result;
    }
}
