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

package org.eclipse.deeplearning4j.vlm.model;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Fixes decoder graph wiring for VLM models imported from ONNX.
 *
 * Handles rewiring inputs_embeds into the decoder graph, fixing repeat_kv
 * reshape operations, and managing causal mask placeholders.
 */
@Slf4j
public class DecoderGraphFixer {

    /**
     * Ensure inputs_embeds is wired into the decoder graph.
     *
     * Strategy: find the layernorm's first input and trace back to find the op
     * that produces it. Then replace that op's embedding input with inputs_embeds,
     * preserving any intermediate operations (like divide/scale).
     *
     * Original path:  input_ids -&gt; embed_lookup -&gt; divide -&gt; layernorm
     * Desired path:   inputs_embeds -&gt; divide -&gt; layernorm
     *
     * @param decoder the SameDiff decoder model
     */
    public static void fixDecoderInputsEmbeds(SameDiff decoder) {
        String targetOpName = null;
        for (String opName : decoder.getOps().keySet()) {
            if (opName.contains("/model/layers.0/input_layernorm") && opName.contains("LayerNorm")) {
                targetOpName = opName;
                break;
            }
        }
        if (targetOpName == null) {
            for (String opName : decoder.getOps().keySet()) {
                if (opName.contains("input_layernorm")) {
                    targetOpName = opName;
                    break;
                }
            }
        }
        if (targetOpName == null) {
            log.warn("Could not locate input_layernorm op to wire inputs_embeds");
            return;
        }

        SameDiffOp layernormOp = decoder.getOps().get(targetOpName);
        if (layernormOp == null || layernormOp.getInputsToOp() == null || layernormOp.getInputsToOp().isEmpty()) {
            log.warn("input_layernorm op '{}' has no inputs", targetOpName);
            return;
        }

        String layernormInput = layernormOp.getInputsToOp().get(0);
        log.info("Layernorm '{}' takes input[0] = '{}'", targetOpName, layernormInput);

        if ("inputs_embeds".equals(layernormInput)) {
            log.info("inputs_embeds already wired into '{}'", targetOpName);
            return;
        }

        Variable upstreamVar = decoder.getVariables().get(layernormInput);
        if (upstreamVar == null) {
            log.warn("Variable '{}' not found, falling back to direct wiring", layernormInput);
            rewireOpInput(decoder, targetOpName, 0, layernormInput, "inputs_embeds");
            return;
        }

        String producerOpName = upstreamVar.getOutputOfOp();
        log.info("Variable '{}' is produced by op '{}'", layernormInput, producerOpName);

        if (producerOpName == null) {
            log.info("'{}' is not produced by any op (type={}), wiring inputs_embeds directly to layernorm",
                    layernormInput, upstreamVar.getVariable().getVariableType());
            rewireOpInput(decoder, targetOpName, 0, layernormInput, "inputs_embeds");
            return;
        }

        SameDiffOp producerOp = decoder.getOps().get(producerOpName);
        if (producerOp == null) {
            log.warn("Producer op '{}' not found, falling back to direct wiring", producerOpName);
            rewireOpInput(decoder, targetOpName, 0, layernormInput, "inputs_embeds");
            return;
        }

        log.info("Producer op '{}' type={}, inputs={}, outputs={}",
                producerOpName,
                producerOp.getOp() != null ? producerOp.getOp().getClass().getSimpleName() : "null",
                producerOp.getInputsToOp(),
                producerOp.getOutputsOfOp());

        if (producerOp.getInputsToOp() != null) {
            for (int i = 0; i < producerOp.getInputsToOp().size(); i++) {
                String inputName = producerOp.getInputsToOp().get(i);
                Variable inputVariable = decoder.getVariables().get(inputName);
                if (inputVariable != null) {
                    SDVariable sdVar = inputVariable.getVariable();
                    log.info("  Producer input[{}] '{}': type={}, outputOfOp={}",
                            i, inputName, sdVar.getVariableType(), inputVariable.getOutputOfOp());
                    if (sdVar.getVariableType() == VariableType.CONSTANT) {
                        INDArray arr = sdVar.getArr();
                        if (arr != null && arr.length() <= 10) {
                            log.info("    Constant value: {}", arr);
                        } else if (arr != null) {
                            log.info("    Constant shape: {}, dtype: {}", java.util.Arrays.toString(arr.shape()), arr.dataType());
                        }
                    }
                }
            }
        }

        List<String> producerInputs = producerOp.getInputsToOp();
        if (producerInputs == null || producerInputs.isEmpty()) {
            log.warn("Producer op '{}' has no inputs, falling back to direct wiring", producerOpName);
            rewireOpInput(decoder, targetOpName, 0, layernormInput, "inputs_embeds");
            return;
        }

        int embedInputIdx = -1;
        for (int i = 0; i < producerInputs.size(); i++) {
            String inputName = producerInputs.get(i);
            Variable v = decoder.getVariables().get(inputName);
            if (v != null && v.getVariable().getVariableType() != VariableType.CONSTANT) {
                embedInputIdx = i;
                log.info("Identified embedding input at producer input[{}] = '{}'", i, inputName);
                break;
            }
        }

        if (embedInputIdx == -1) {
            log.warn("No non-constant input found in producer op '{}', falling back to direct wiring", producerOpName);
            rewireOpInput(decoder, targetOpName, 0, layernormInput, "inputs_embeds");
            return;
        }

        for (int i = 0; i < producerInputs.size(); i++) {
            String inputName = producerInputs.get(i);
            if (i != embedInputIdx) {
                log.info("=== Tracing computation chain for producer input[{}] '{}' (depth=5) ===", i, inputName);
                traceGraphUpstream(decoder, inputName, 5);
            }
        }

        String oldEmbedInput = producerInputs.get(embedInputIdx);
        log.info("Rewiring producer op '{}' input[{}] from '{}' to 'inputs_embeds' (preserving {} operation)",
                producerOpName, embedInputIdx, oldEmbedInput, producerOpName);
        rewireOpInput(decoder, producerOpName, embedInputIdx, oldEmbedInput, "inputs_embeds");
    }

    /**
     * Diagnose and log the repeat_kv Reshape operations in the decoder model.
     *
     * @param decoder the SameDiff decoder model
     */
    public static void fixRepeatKVReshape(SameDiff decoder) {
        log.info("=== Checking repeat_kv Reshape operations ===");

        for (String opName : decoder.getOps().keySet()) {
            if (opName.contains("repeat_kv") && opName.contains("Reshape_1")) {
                SameDiffOp op = decoder.getOps().get(opName);
                log.info("Found Reshape: {}", opName);
                log.info("  Inputs: {}", op.getInputsToOp());
                log.info("  Outputs: {}", op.getOutputsOfOp());

                if (op.getInputsToOp().size() >= 2) {
                    String shapeInputName = op.getInputsToOp().get(1);
                    SDVariable shapeVar = decoder.getVariable(shapeInputName);
                    if (shapeVar != null) {
                        log.info("  Shape variable: {}", shapeInputName);
                        log.info("  Shape var type: {}", shapeVar.getVariableType());
                        INDArray shapeArr = shapeVar.getArr();
                        if (shapeArr != null) {
                            log.info("  Shape values: {}", shapeArr);
                        } else {
                            log.info("  Shape array is null (computed at runtime)");
                        }
                    }
                }

                break;
            }
        }

        log.info("\n=== Checking shape constants ===");
        for (String varName : decoder.getVariables().keySet()) {
            if (varName.contains("0, 0, 3, -1") || varName.contains("constants") && varName.contains("INT64")) {
                SDVariable var = decoder.getVariable(varName);
                log.info("Constant: {}", varName);
                log.info("  Type: {}", var.getVariableType());
                INDArray arr = var.getArr();
                if (arr != null) {
                    log.info("  Values: {}", arr);
                }
            }
        }

        log.info("=== repeat_kv check complete ===");
    }

    /**
     * Fix the decoder model's baked-in input_ids constant by creating a
     * _causal_mask FLOAT placeholder and wiring it to the Tile operation.
     *
     * @param decoder the SameDiff decoder model
     */
    public static void fixDecoderInputIds(SameDiff decoder) {
        log.info("=== Fixing decoder input_ids (causal mask) ===");

        String tileOpName = "/model/attn_mask_reformat/Tile";
        SameDiffOp tileOp = decoder.getOps().get(tileOpName);
        if (tileOp == null) {
            log.error("Could not find Tile operation at {}", tileOpName);
            return;
        }

        String addOpName = "/model/attn_mask_reformat/Add";
        SameDiffOp addOp = decoder.getOps().get(addOpName);
        String addOutputName = null;
        if (addOp != null) {
            addOutputName = addOp.getOutputsOfOp().get(0);
            log.info("Found Add operation: {}, output: {}", addOpName, addOutputName);
        }

        log.info("Tile operation inputs: {}", tileOp.getInputsToOp());

        SDVariable causalMask = decoder.placeHolder("_causal_mask", DataType.FLOAT, -1, -1, -1, -1);
        log.info("Created _causal_mask FLOAT placeholder for causal attention masking");

        List<String> tileInputs = new ArrayList<>(tileOp.getInputsToOp());
        String oldTileInput = tileInputs.get(0);
        log.info("Replacing Tile input[0] '{}' with '{}'", oldTileInput, causalMask.name());
        tileInputs.set(0, causalMask.name());
        tileOp.setInputsToOp(tileInputs);

        Variable causalVar = decoder.getVariables().get(causalMask.name());
        if (causalVar != null) {
            List<String> inputsForOp = causalVar.getInputsForOp();
            if (inputsForOp == null) {
                inputsForOp = new ArrayList<>();
                causalVar.setInputsForOp(inputsForOp);
            }
            inputsForOp.add(tileOpName);
        }

        if (addOutputName != null) {
            Variable oldVar = decoder.getVariables().get(addOutputName);
            if (oldVar != null && oldVar.getInputsForOp() != null) {
                oldVar.getInputsForOp().remove(tileOpName);
            }
        }

        log.info("=== Fix complete ===");
        log.info("Tile operation now has inputs: {}", tileOp.getInputsToOp());

        List<String> allInputs = new ArrayList<>();
        for (String varName : decoder.getVariables().keySet()) {
            Variable v = decoder.getVariables().get(varName);
            if (v.getVariable() != null && v.getVariable().getVariableType() == VariableType.PLACEHOLDER) {
                allInputs.add(varName);
            }
        }
        Collections.sort(allInputs);
        log.info("Fixed decoder input_ids: now has inputs={}", allInputs);
    }

    /**
     * Rewire a specific input of an op from one variable to another,
     * updating all variable tracking metadata.
     *
     * @param decoder the SameDiff model
     * @param opName the op to modify
     * @param inputIdx the input index to change
     * @param oldVarName the old variable name
     * @param newVarName the new variable name
     */
    public static void rewireOpInput(SameDiff decoder, String opName, int inputIdx, String oldVarName, String newVarName) {
        SameDiffOp op = decoder.getOps().get(opName);
        if (op == null) {
            log.error("Cannot rewire: op '{}' not found", opName);
            return;
        }

        List<String> inputs = new ArrayList<>(op.getInputsToOp());
        inputs.set(inputIdx, newVarName);
        op.setInputsToOp(inputs);

        Variable newVar = decoder.getVariables().get(newVarName);
        if (newVar != null) {
            List<String> inputsForOp = newVar.getInputsForOp();
            if (inputsForOp == null) {
                inputsForOp = new ArrayList<>();
                newVar.setInputsForOp(inputsForOp);
            }
            if (!inputsForOp.contains(opName)) {
                inputsForOp.add(opName);
            }
        }

        Variable oldVar = decoder.getVariables().get(oldVarName);
        if (oldVar != null && oldVar.getInputsForOp() != null) {
            oldVar.getInputsForOp().remove(opName);
        }

        log.info("Rewired '{}' input[{}]: '{}' -> '{}'", opName, inputIdx, oldVarName, newVarName);
    }

    /**
     * Trace the computation graph upstream from a variable, logging the ops and inputs.
     *
     * @param graph the SameDiff graph
     * @param varName the variable to trace from
     * @param maxDepth maximum trace depth
     */
    public static void traceGraphUpstream(SameDiff graph, String varName, int maxDepth) {
        traceGraphUpstream(graph, varName, maxDepth, 0);
    }

    private static void traceGraphUpstream(SameDiff graph, String varName, int maxDepth, int depth) {
        if (depth >= maxDepth) return;
        String indent = "  ".repeat(depth);

        Variable v = graph.getVariables().get(varName);
        if (v == null) {
            log.info("{}[trace] '{}': NOT FOUND", indent, varName);
            return;
        }

        SDVariable sdVar = v.getVariable();
        String producerOp = v.getOutputOfOp();
        log.info("{}[trace] '{}': type={}, outputOfOp={}", indent, varName, sdVar.getVariableType(), producerOp);

        if (sdVar.getVariableType() == VariableType.CONSTANT) {
            INDArray arr = sdVar.getArr();
            if (arr != null) {
                if (arr.length() <= 10) {
                    log.info("{}  constant value: {}", indent, arr);
                } else {
                    log.info("{}  constant shape={}, dtype={}", indent,
                            java.util.Arrays.toString(arr.shape()), arr.dataType());
                }
            }
            return;
        }

        if (producerOp == null) return;

        SameDiffOp op = graph.getOps().get(producerOp);
        if (op == null) {
            log.info("{}  op '{}' NOT FOUND", indent, producerOp);
            return;
        }

        String opType = op.getOp() != null ? op.getOp().getClass().getSimpleName() : "null";
        log.info("{}  op '{}' type={}, inputs={}", indent, producerOp, opType, op.getInputsToOp());

        if (op.getInputsToOp() != null) {
            for (String inputName : op.getInputsToOp()) {
                traceGraphUpstream(graph, inputName, maxDepth, depth + 1);
            }
        }
    }
}
