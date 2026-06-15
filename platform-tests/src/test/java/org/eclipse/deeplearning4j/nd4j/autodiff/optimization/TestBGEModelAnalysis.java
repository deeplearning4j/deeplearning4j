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

package org.eclipse.deeplearning4j.nd4j.autodiff.optimization;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.ops.impl.reduce.TensorMmul;
import org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax;

import java.io.File;
import java.util.*;

/**
 * Test to analyze the BGE model structure and understand attention patterns.
 */
public class TestBGEModelAnalysis {

    private static final String MODEL_PATH = System.getProperty("bge.model.path",
            System.getProperty("user.home") + "/.kompile/models/encoders/bge-base-en-v1.5/model-unoptimized.sdz");

    @Test
    public void analyzeModelStructure() throws Exception {
        File modelFile = new File(MODEL_PATH);
        assumeTrue(modelFile.exists(), "BGE model not found at " + MODEL_PATH
                + ". Set -Dbge.model.path to provide the model.");

        System.out.println("Loading model from: " + MODEL_PATH);
        SameDiff sd = SameDiff.loadSharded(modelFile);

        System.out.println("\n=== MODEL SUMMARY ===");
        System.out.println("Total ops: " + sd.getOps().size());
        System.out.println("Total variables: " + sd.getVariables().size());

        // Count ops by type
        Map<String, Integer> opCounts = new TreeMap<>();
        for (SameDiffOp op : sd.getOps().values()) {
            String opType = op.getOp().getClass().getSimpleName();
            opCounts.merge(opType, 1, Integer::sum);
        }

        System.out.println("\n=== OP COUNTS BY TYPE ===");
        opCounts.entrySet().stream()
            .sorted((a, b) -> b.getValue().compareTo(a.getValue()))
            .forEach(e -> System.out.println(e.getKey() + ": " + e.getValue()));

        // Find all softmax ops and analyze their inputs/outputs
        System.out.println("\n=== SOFTMAX ANALYSIS ===");
        int softmaxCount = 0;
        for (SameDiffOp op : sd.getOps().values()) {
            if (op.getOp() instanceof SoftMax) {
                softmaxCount++;
                if (softmaxCount <= 3) { // Only print first 3
                    analyzeSoftmaxOp(sd, op);
                }
            }
        }
        System.out.println("Total softmax ops: " + softmaxCount);

        // Find matmul ops that might be attention
        System.out.println("\n=== MATMUL/TENSORMMUL ANALYSIS ===");
        int matmulCount = 0;
        for (SameDiffOp op : sd.getOps().values()) {
            if (op.getOp() instanceof Mmul || op.getOp() instanceof TensorMmul) {
                matmulCount++;
                if (matmulCount <= 5) { // Only print first 5
                    analyzeMatmulOp(sd, op);
                }
            }
        }
        System.out.println("Total matmul/tensormmul ops: " + matmulCount);

        // Try to find attention patterns by looking at softmax consumers
        System.out.println("\n=== ATTENTION PATTERN SEARCH ===");
        findAttentionPatterns(sd);
    }

    private void analyzeSoftmaxOp(SameDiff sd, SameDiffOp softmaxOp) {
        System.out.println("\nSoftmax op: " + softmaxOp.getName());

        // Inputs
        List<String> inputs = softmaxOp.getInputsToOp();
        System.out.println("  Inputs: " + inputs);
        if (inputs != null) {
            for (String input : inputs) {
                Variable v = sd.getVariables().get(input);
                if (v != null && v.getOutputOfOp() != null) {
                    SameDiffOp producer = sd.getOps().get(v.getOutputOfOp());
                    if (producer != null) {
                        System.out.println("    " + input + " <- " + producer.getName() +
                            " (" + producer.getOp().getClass().getSimpleName() + ")");
                        // Go one level deeper
                        List<String> producerInputs = producer.getInputsToOp();
                        if (producerInputs != null) {
                            for (String pi : producerInputs) {
                                Variable pv = sd.getVariables().get(pi);
                                if (pv != null && pv.getOutputOfOp() != null) {
                                    SameDiffOp pp = sd.getOps().get(pv.getOutputOfOp());
                                    if (pp != null) {
                                        System.out.println("      " + pi + " <- " + pp.getName() +
                                            " (" + pp.getOp().getClass().getSimpleName() + ")");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Outputs
        List<String> outputs = softmaxOp.getOutputsOfOp();
        System.out.println("  Outputs: " + outputs);
        if (outputs != null) {
            for (String output : outputs) {
                Variable v = sd.getVariables().get(output);
                if (v != null) {
                    List<String> consumers = v.getInputsForOp();
                    if (consumers != null) {
                        for (String consumer : consumers) {
                            SameDiffOp consumerOp = sd.getOps().get(consumer);
                            if (consumerOp != null) {
                                System.out.println("    " + output + " -> " + consumer +
                                    " (" + consumerOp.getOp().getClass().getSimpleName() + ")");
                            }
                        }
                    }
                }
            }
        }
    }

    private void analyzeMatmulOp(SameDiff sd, SameDiffOp matmulOp) {
        System.out.println("\nMatmul op: " + matmulOp.getName() + " (" + matmulOp.getOp().getClass().getSimpleName() + ")");

        // Inputs
        List<String> inputs = matmulOp.getInputsToOp();
        System.out.println("  Inputs: " + inputs);
        if (inputs != null) {
            for (String input : inputs) {
                Variable v = sd.getVariables().get(input);
                if (v != null && v.getOutputOfOp() != null) {
                    SameDiffOp producer = sd.getOps().get(v.getOutputOfOp());
                    if (producer != null) {
                        System.out.println("    " + input + " <- " + producer.getName() +
                            " (" + producer.getOp().getClass().getSimpleName() + ")");
                    }
                } else if (v != null) {
                    System.out.println("    " + input + " <- [INPUT/CONSTANT]");
                }
            }
        }

        // Outputs and consumers
        List<String> outputs = matmulOp.getOutputsOfOp();
        if (outputs != null) {
            for (String output : outputs) {
                Variable v = sd.getVariables().get(output);
                if (v != null) {
                    List<String> consumers = v.getInputsForOp();
                    if (consumers != null) {
                        for (String consumer : consumers) {
                            SameDiffOp consumerOp = sd.getOps().get(consumer);
                            if (consumerOp != null) {
                                System.out.println("    " + output + " -> " + consumer +
                                    " (" + consumerOp.getOp().getClass().getSimpleName() + ")");
                            }
                        }
                    }
                }
            }
        }
    }

    private void findAttentionPatterns(SameDiff sd) {
        int patternsFound = 0;

        // For each matmul, trace back to see if it's an attention pattern
        for (SameDiffOp op : sd.getOps().values()) {
            if (!(op.getOp() instanceof Mmul || op.getOp() instanceof TensorMmul)) continue;

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() < 2) continue;

            // Check if either input traces back to softmax
            for (int i = 0; i < 2; i++) {
                String input = inputs.get(i);
                String softmaxFound = traceToSoftmax(sd, input, 10);
                if (softmaxFound != null) {
                    patternsFound++;
                    if (patternsFound <= 3) {
                        System.out.println("\nPotential attention pattern #" + patternsFound);
                        System.out.println("  Final matmul: " + op.getName());
                        System.out.println("  Input " + i + " traces to softmax: " + softmaxFound);
                        System.out.println("  Tracing path from " + input + ":");
                        traceBackFromSoftmaxInput(sd, input, "    ");
                    }
                    break;
                }
            }
        }
        System.out.println("\nTotal potential attention patterns found: " + patternsFound);
    }

    private String traceToSoftmax(SameDiff sd, String varName, int maxDepth) {
        if (maxDepth <= 0) return null;

        Variable v = sd.getVariables().get(varName);
        if (v == null) return null;

        String producerName = v.getOutputOfOp();
        if (producerName == null) return null;

        SameDiffOp producer = sd.getOps().get(producerName);
        if (producer == null) return null;

        if (producer.getOp() instanceof SoftMax) {
            return producerName;
        }

        // Continue tracing through reshape, permute, transpose
        if (producer.getOp() instanceof org.nd4j.linalg.api.ops.impl.shape.Reshape ||
            producer.getOp() instanceof org.nd4j.linalg.api.ops.impl.shape.Permute ||
            producer.getOp() instanceof org.nd4j.linalg.api.ops.impl.shape.Transpose) {
            List<String> producerInputs = producer.getInputsToOp();
            if (producerInputs != null && !producerInputs.isEmpty()) {
                return traceToSoftmax(sd, producerInputs.get(0), maxDepth - 1);
            }
        }

        return null;
    }

    @Test
    public void testManualAttentionFusion() throws Exception {
        File modelFile = new File(MODEL_PATH);
        assumeTrue(modelFile.exists(), "BGE model not found at " + MODEL_PATH
                + ". Set -Dbge.model.path to provide the model.");

        System.out.println("Loading unoptimized model...");
        SameDiff sd = SameDiff.loadSharded(modelFile);

        // Find matmul_1 and manually trace the pattern
        SameDiffOp matmul1 = sd.getOps().get("matmul_1");
        if (matmul1 == null) {
            System.out.println("matmul_1 not found!");
            return;
        }

        System.out.println("\n=== Manual pattern check for matmul_1 ===");
        List<String> inputs = matmul1.getInputsToOp();
        System.out.println("matmul_1 inputs: " + inputs);

        // First input should be attention weights (from softmax)
        String input0 = inputs.get(0);
        System.out.println("\nTracing input 0: " + input0);

        // Trace through reshapes to find softmax
        String current = input0;
        int depth = 0;
        while (depth < 5) {
            Variable v = sd.getVariables().get(current);
            if (v == null) {
                System.out.println("  Variable " + current + " not found!");
                break;
            }

            String producerName = v.getOutputOfOp();
            if (producerName == null) {
                System.out.println("  " + current + " is a placeholder/constant");
                break;
            }

            SameDiffOp producer = sd.getOps().get(producerName);
            String opType = producer.getOp().getClass().getSimpleName();
            System.out.println("  " + current + " <- " + producerName + " (" + opType + ")");

            // Check number of users
            List<String> users = v.getInputsForOp();
            System.out.println("    Users of " + current + ": " + (users != null ? users.size() : 0) + " -> " + users);

            if (producer.getOp() instanceof SoftMax) {
                System.out.println("  >>> Found softmax!");
                // Now trace back from softmax input
                List<String> softmaxInputs = producer.getInputsToOp();
                System.out.println("  Softmax inputs: " + softmaxInputs);
                break;
            }

            // Continue tracing through reshape/permute
            if (producer.getOp() instanceof org.nd4j.linalg.api.ops.impl.shape.Reshape ||
                producer.getOp() instanceof org.nd4j.linalg.api.ops.impl.shape.Permute ||
                producer.getOp() instanceof org.nd4j.linalg.api.ops.impl.shape.Transpose) {
                List<String> producerInputs = producer.getInputsToOp();
                if (producerInputs != null && !producerInputs.isEmpty()) {
                    current = producerInputs.get(0);
                    depth++;
                } else {
                    break;
                }
            } else {
                System.out.println("  Unexpected op type: " + opType);
                break;
            }
        }

        // Check Q@K^T matmul
        SameDiffOp qkMatmul = sd.getOps().get("matmul");
        if (qkMatmul != null) {
            System.out.println("\n=== Q@K^T matmul check ===");
            System.out.println("matmul inputs: " + qkMatmul.getInputsToOp());
            List<String> qkOutputs = qkMatmul.getOutputsOfOp();
            System.out.println("matmul outputs: " + qkOutputs);

            if (qkOutputs != null && !qkOutputs.isEmpty()) {
                Variable qkOutVar = sd.getVariables().get(qkOutputs.get(0));
                if (qkOutVar != null) {
                    List<String> qkUsers = qkOutVar.getInputsForOp();
                    System.out.println("matmul output users: " + qkUsers.size() + " -> " + qkUsers);
                }
            }
        }

        // Trace from softmax input backwards to Q@K^T matmul
        System.out.println("\n=== Full trace from softmax input to Q@K^T ===");
        String softmaxInputVar = "Attention_0_scaledScoresWithMask";
        traceVariableBackwards(sd, softmaxInputVar, "  ", 10);
    }

    private void traceVariableBackwards(SameDiff sd, String varName, String indent, int maxDepth) {
        if (maxDepth <= 0) {
            System.out.println(indent + varName + " ... (max depth reached)");
            return;
        }

        Variable v = sd.getVariables().get(varName);
        if (v == null) {
            System.out.println(indent + varName + " <- [NOT FOUND]");
            return;
        }

        String producerName = v.getOutputOfOp();
        if (producerName == null) {
            System.out.println(indent + varName + " <- [INPUT/CONSTANT]");
            return;
        }

        SameDiffOp producer = sd.getOps().get(producerName);
        if (producer == null) {
            System.out.println(indent + varName + " <- [PRODUCER NOT FOUND: " + producerName + "]");
            return;
        }

        String opType = producer.getOp().getClass().getSimpleName();
        List<String> users = v.getInputsForOp();
        System.out.println(indent + varName + " <- " + producerName + " (" + opType + ") [users: " + (users != null ? users.size() : 0) + "]");

        // Trace all inputs of this op
        List<String> inputs = producer.getInputsToOp();
        if (inputs != null) {
            for (String input : inputs) {
                traceVariableBackwards(sd, input, indent + "  ", maxDepth - 1);
            }
        }
    }

    @Test
    public void testOptimizerOnBGE() throws Exception {
        File modelFile = new File(MODEL_PATH);
        assumeTrue(modelFile.exists(), "BGE model not found at " + MODEL_PATH
                + ". Set -Dbge.model.path to provide the model.");

        System.out.println("Loading unoptimized model...");
        SameDiff sd = SameDiff.loadSharded(modelFile);

        // Count ops before optimization
        int softmaxBefore = 0, matmulBefore = 0, attentionV2Before = 0;
        for (SameDiffOp op : sd.getOps().values()) {
            if (op.getOp() instanceof SoftMax) softmaxBefore++;
            if (op.getOp() instanceof Mmul || op.getOp() instanceof TensorMmul) matmulBefore++;
            if (op.getOp() instanceof DotProductAttentionV2) attentionV2Before++;
        }

        System.out.println("\n=== BEFORE OPTIMIZATION ===");
        System.out.println("Softmax ops: " + softmaxBefore);
        System.out.println("Matmul/TensorMmul ops: " + matmulBefore);
        System.out.println("DotProductAttentionV2 ops: " + attentionV2Before);

        // Find the output variable
        String outputVar = "last_hidden_state";
        if (!sd.getVariables().containsKey(outputVar)) {
            // Try to find an output
            for (String var : sd.getVariables().keySet()) {
                Variable v = sd.getVariables().get(var);
                if (v.getInputsForOp() == null || v.getInputsForOp().isEmpty()) {
                    if (v.getOutputOfOp() != null) {
                        outputVar = var;
                        break;
                    }
                }
            }
        }
        System.out.println("Using output variable: " + outputVar);

        // Run optimizer
        System.out.println("\nRunning graph optimizer...");
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList(outputVar));

        // Count ops after optimization
        int softmaxAfter = 0, matmulAfter = 0, attentionV2After = 0;
        for (SameDiffOp op : optimized.getOps().values()) {
            if (op.getOp() instanceof SoftMax) softmaxAfter++;
            if (op.getOp() instanceof Mmul || op.getOp() instanceof TensorMmul) matmulAfter++;
            if (op.getOp() instanceof DotProductAttentionV2) attentionV2After++;
        }

        System.out.println("\n=== AFTER OPTIMIZATION ===");
        System.out.println("Softmax ops: " + softmaxAfter);
        System.out.println("Matmul/TensorMmul ops: " + matmulAfter);
        System.out.println("DotProductAttentionV2 ops: " + attentionV2After);

        System.out.println("\n=== CHANGES ===");
        System.out.println("Softmax: " + softmaxBefore + " -> " + softmaxAfter + " (change: " + (softmaxAfter - softmaxBefore) + ")");
        System.out.println("Matmul: " + matmulBefore + " -> " + matmulAfter + " (change: " + (matmulAfter - matmulBefore) + ")");
        System.out.println("DotProductAttentionV2: " + attentionV2Before + " -> " + attentionV2After + " (change: " + (attentionV2After - attentionV2Before) + ")");

        if (attentionV2After > attentionV2Before) {
            System.out.println("\n*** ATTENTION FUSION APPLIED! ***");
        } else {
            System.out.println("\n*** ATTENTION FUSION NOT APPLIED ***");
        }

        System.out.println();
    }

    private void traceBackFromSoftmaxInput(SameDiff sd, String varName, String indent) {
        Variable v = sd.getVariables().get(varName);
        if (v == null) return;

        String producerName = v.getOutputOfOp();
        if (producerName == null) {
            System.out.println(indent + varName + " <- [INPUT/CONSTANT]");
            return;
        }

        SameDiffOp producer = sd.getOps().get(producerName);
        if (producer == null) return;

        String opType = producer.getOp().getClass().getSimpleName();
        System.out.println(indent + varName + " <- " + producerName + " (" + opType + ")");

        // Continue tracing for a few levels
        if (indent.length() < 16) {
            List<String> inputs = producer.getInputsToOp();
            if (inputs != null) {
                for (String input : inputs) {
                    traceBackFromSoftmaxInput(sd, input, indent + "  ");
                }
            }
        }
    }
}
