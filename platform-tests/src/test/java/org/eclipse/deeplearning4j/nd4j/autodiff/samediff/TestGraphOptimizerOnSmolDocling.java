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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;
import org.nd4j.autodiff.samediff.optimize.optimizations.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Transparent test for GraphOptimizer applied to real ONNX models.
 *
 * This test downloads the SmolDocling decoder/vision/embed models,
 * runs GraphOptimizer with full logging, and reports exactly which
 * optimizations were applied, how many ops were removed, and what
 * the op type distribution looks like before vs. after.
 *
 * Run:
 *   cd platform-tests && mvn test \
 *     -Dtest=TestGraphOptimizerOnSmolDocling#testOptimizerOnDecoder \
 *     -Dnd4j.optimizer.logApplied=true
 *
 * The logApplied flag causes GraphOptimizer to log each individual
 * optimization application (op name + optimizer class).
 */
@Slf4j
public class TestGraphOptimizerOnSmolDocling {

    /**
     * Apply GraphOptimizer to the SmolDocling decoder and report results.
     */
    @Test
    @DisplayName("GraphOptimizer on SmolDocling decoder: transparent report")
    public void testOptimizerOnDecoder() throws Exception {
        // Download decoder model
        log.info("=== GraphOptimizer Transparency Test: SmolDocling Decoder ===");
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        assertTrue(decoderResult.getModelFile().exists(), "Decoder model file should exist");

        // Import from ONNX (raw, no caching, no optimization)
        log.info("Importing ONNX model (raw, no optimization)...");
        long importStart = System.currentTimeMillis();
        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff original = importer.runImport(decoderResult.getModelFile().getAbsolutePath(), Map.of(), false, false);
        long importTime = System.currentTimeMillis() - importStart;
        log.info("Import completed in {}ms", importTime);

        // Report BEFORE state
        log.info("");
        log.info("========== BEFORE OPTIMIZATION ==========");
        Map<String, Integer> opsBefore = countOpsByType(original);
        int totalOpsBefore = original.getOps().size();
        Map<VariableType, Integer> varsBefore = countVarsByType(original);
        Map<DataType, Integer> dtypesBefore = countConstantsByDtype(original);

        log.info("Total ops: {}", totalOpsBefore);
        log.info("Total variables: {}", original.getVariables().size());
        log.info("Outputs: {}", original.outputs());
        log.info("");
        log.info("Op type distribution (top 20):");
        printOpDistribution(opsBefore, 20);
        log.info("");
        log.info("Variable types: {}", varsBefore);
        log.info("Constant dtypes: {}", dtypesBefore);

        // Run optimizer with each pass individually for transparency
        List<String> outputs = original.outputs() != null ? new ArrayList<>(original.outputs()) : new ArrayList<>();
        assertFalse(outputs.isEmpty(), "Model should have outputs defined");

        log.info("");
        log.info("========== RUNNING INDIVIDUAL OPTIMIZATION PASSES ==========");

        // Run each pass individually to see what each one does
        List<OptimizerSet> allPasses = GraphOptimizer.defaultOptimizations();
        log.info("Default optimization passes ({}):", allPasses.size());
        for (int i = 0; i < allPasses.size(); i++) {
            log.info("  [{}] {}", i, allPasses.get(i).getClass().getSimpleName());
        }

        // Now run the full optimizer
        log.info("");
        log.info("========== RUNNING FULL GRAPHOPTIMIZER ==========");
        long optStart = System.currentTimeMillis();
        SameDiff optimized = GraphOptimizer.optimize(original, outputs);
        long optTime = System.currentTimeMillis() - optStart;

        // Report AFTER state
        log.info("");
        log.info("========== AFTER OPTIMIZATION ==========");
        Map<String, Integer> opsAfter = countOpsByType(optimized);
        int totalOpsAfter = optimized.getOps().size();
        Map<VariableType, Integer> varsAfter = countVarsByType(optimized);
        Map<DataType, Integer> dtypesAfter = countConstantsByDtype(optimized);

        log.info("Total ops: {} (was {}, removed {})", totalOpsAfter, totalOpsBefore, totalOpsBefore - totalOpsAfter);
        log.info("Total variables: {} (was {})", optimized.getVariables().size(), original.getVariables().size());
        log.info("Optimization time: {}ms", optTime);
        log.info("");
        log.info("Op type distribution (top 20):");
        printOpDistribution(opsAfter, 20);
        log.info("");
        log.info("Variable types: {}", varsAfter);
        log.info("Constant dtypes: {}", dtypesAfter);

        // Show diff: which op types changed
        log.info("");
        log.info("========== OP TYPE CHANGES ==========");
        Set<String> allOpTypes = new TreeSet<>();
        allOpTypes.addAll(opsBefore.keySet());
        allOpTypes.addAll(opsAfter.keySet());

        for (String opType : allOpTypes) {
            int before = opsBefore.getOrDefault(opType, 0);
            int after = opsAfter.getOrDefault(opType, 0);
            if (before != after) {
                String delta = after > before ? "+" + (after - before) : String.valueOf(after - before);
                log.info("  {}: {} -> {} ({})", opType, before, after, delta);
            }
        }

        // Show new op types (fused ops introduced by optimizer)
        Set<String> newOpTypes = new TreeSet<>(opsAfter.keySet());
        newOpTypes.removeAll(opsBefore.keySet());
        if (!newOpTypes.isEmpty()) {
            log.info("");
            log.info("New op types introduced by optimization:");
            for (String opType : newOpTypes) {
                log.info("  {} ({}x)", opType, opsAfter.get(opType));
            }
        }

        // Show removed op types
        Set<String> removedOpTypes = new TreeSet<>(opsBefore.keySet());
        removedOpTypes.removeAll(opsAfter.keySet());
        if (!removedOpTypes.isEmpty()) {
            log.info("");
            log.info("Op types completely eliminated:");
            for (String opType : removedOpTypes) {
                log.info("  {} (was {}x)", opType, opsBefore.get(opType));
            }
        }

        // Summary
        log.info("");
        log.info("========== SUMMARY ==========");
        double pctReduction = totalOpsBefore > 0 ? (100.0 * (totalOpsBefore - totalOpsAfter) / totalOpsBefore) : 0;
        log.info("Model: SmolDocling decoder");
        log.info("Ops: {} -> {} ({}% reduction)", totalOpsBefore, totalOpsAfter, String.format("%.1f", pctReduction));
        log.info("Variables: {} -> {}", original.getVariables().size(), optimized.getVariables().size());
        log.info("Optimization time: {}ms", optTime);
        log.info("Outputs preserved: {}", optimized.outputs());

        // Basic sanity: optimized graph should have outputs and ops
        assertFalse(optimized.getOps().isEmpty(), "Optimized graph should have ops");
        assertNotNull(optimized.outputs(), "Optimized graph should have outputs");
        assertFalse(optimized.outputs().isEmpty(), "Optimized graph should have non-empty outputs");
    }

    /**
     * Apply GraphOptimizer to the SmolDocling vision encoder and report results.
     */
    @Test
    @DisplayName("GraphOptimizer on SmolDocling vision encoder: transparent report")
    public void testOptimizerOnVisionEncoder() throws Exception {
        log.info("=== GraphOptimizer Transparency Test: SmolDocling Vision Encoder ===");
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        assertTrue(visionResult.getModelFile().exists());

        log.info("Importing ONNX model...");
        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff original = importer.runImport(visionResult.getModelFile().getAbsolutePath(), Map.of(), false, false);

        reportOptimization(original, "SmolDocling Vision Encoder");
    }

    /**
     * Apply GraphOptimizer to the SmolDocling embed_tokens model and report results.
     */
    @Test
    @DisplayName("GraphOptimizer on SmolDocling embed_tokens: transparent report")
    public void testOptimizerOnEmbedTokens() throws Exception {
        log.info("=== GraphOptimizer Transparency Test: SmolDocling embed_tokens ===");
        var embedResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        assertTrue(embedResult.getModelFile().exists());

        log.info("Importing ONNX model...");
        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff original = importer.runImport(embedResult.getModelFile().getAbsolutePath(), Map.of(), false, false);

        reportOptimization(original, "SmolDocling embed_tokens");
    }

    /**
     * Inspect the graph structure of RMSNorm patterns in the decoder.
     * This traces the pattern from the weight multiply backwards to understand
     * what ops are used and whether the pattern matches expectations.
     */
    @Test
    @DisplayName("Inspect RMSNorm pattern in SmolDocling decoder")
    public void testInspectRMSNormPattern() throws Exception {
        log.info("=== Inspecting RMSNorm Pattern in SmolDocling Decoder ===");
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff sd = importer.runImport(decoderResult.getModelFile().getAbsolutePath(), Map.of(), false, false);

        // Find the first layer's input_layernorm weight variable
        for (SDVariable v : sd.variables()) {
            if (v.name().contains("input_layernorm.weight") && v.name().contains("layers.0")) {
                log.info("Found weight: name={}, type={}, shape={}, dtype={}",
                    v.name(), v.getVariableType(),
                    v.getShape() != null ? java.util.Arrays.toString(v.getShape()) : "null",
                    v.dataType());

                // Find ops that use this weight
                var variable = sd.getVariables().get(v.name());
                if (variable != null) {
                    List<String> usedByOps = variable.getInputsForOp();
                    log.info("  Used by ops: {}", usedByOps);
                    if (usedByOps != null) {
                        for (String opName : usedByOps) {
                            SameDiffOp op = sd.getOps().get(opName);
                            if (op != null) {
                                log.info("  Op: name={}, type={}, opName={}", opName,
                                    op.getOp().getClass().getSimpleName(), op.getOp().opName());
                                log.info("    inputs: {}", op.getInputsToOp());
                                log.info("    outputs: {}", op.getOutputsOfOp());

                                // Trace the other input to this multiply
                                if (op.getInputsToOp() != null) {
                                    for (String input : op.getInputsToOp()) {
                                        if (!input.equals(v.name())) {
                                            SDVariable inputVar = sd.getVariable(input);
                                            log.info("    Other input: name={}, type={}",
                                                input, inputVar != null ? inputVar.getVariableType() : "null");
                                            var inputVariable = sd.getVariables().get(input);
                                            if (inputVariable != null) {
                                                String producerOp = inputVariable.getOutputOfOp();
                                                log.info("    Producer op: {}", producerOp);
                                                if (producerOp != null) {
                                                    SameDiffOp producer = sd.getOps().get(producerOp);
                                                    if (producer != null) {
                                                        log.info("    Producer type: {}, opName={}",
                                                            producer.getOp().getClass().getSimpleName(),
                                                            producer.getOp().opName());
                                                        log.info("    Producer inputs: {}", producer.getInputsToOp());
                                                        // Trace full chain up to 6 levels deep
                                                        traceChain(sd, producer, 0, 6);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Also check the LayerNorm op
                if (variable != null && variable.getInputsForOp() != null) {
                    for (String opName : variable.getInputsForOp()) {
                        SameDiffOp layerNormOp = sd.getOps().get(opName);
                        if (layerNormOp != null && !opName.equals("/model/layers.0/input_layernorm/output_0")) {
                            log.info("  ALSO used by: name={}, type={}, opName={}", opName,
                                layerNormOp.getOp().getClass().getSimpleName(),
                                layerNormOp.getOp().opName());
                            log.info("    inputs: {}", layerNormOp.getInputsToOp());
                            log.info("    outputs: {}", layerNormOp.getOutputsOfOp());
                        }
                    }
                }
                break; // Only inspect the first one
            }
        }
    }

    private void traceChain(SameDiff sd, SameDiffOp op, int depth, int maxDepth) {
        if (depth >= maxDepth || op == null) return;
        String indent = "      " + "  ".repeat(depth);
        log.info("{}op: {} ({}), inputs: {}", indent, op.getName(), op.getOp().opName(), op.getInputsToOp());
        if (op.getInputsToOp() != null) {
            for (String inputName : op.getInputsToOp()) {
                var variable = sd.getVariables().get(inputName);
                SDVariable sdVar = sd.getVariable(inputName);
                String type = sdVar != null ? sdVar.getVariableType().toString() : "?";
                if (variable != null && variable.getOutputOfOp() != null) {
                    SameDiffOp parentOp = sd.getOps().get(variable.getOutputOfOp());
                    log.info("{}  {} [{}] <- {}", indent, inputName, type, variable.getOutputOfOp());
                    traceChain(sd, parentOp, depth + 1, maxDepth);
                } else {
                    long[] shape = sdVar != null ? sdVar.getShape() : null;
                    log.info("{}  {} [{}] shape={}", indent, inputName, type,
                        shape != null ? java.util.Arrays.toString(shape) : "null");
                }
            }
        }
    }

    /**
     * Load the OPTIMIZED SDZ from cache and dump the full graph structure.
     * This shows what the pipeline actually executes — post-optimizer, post-save.
     */
    @Test
    @DisplayName("Dump optimized SDZ graph structure for SmolDocling decoder")
    public void testDumpOptimizedSdzGraph() throws Exception {
        File optSdz = new File(System.getProperty("user.home") + "/.cache/dl4j-vlm-models/smoldocling-decoder.opt.sdz");
        File rawSdz = new File(System.getProperty("user.home") + "/.cache/dl4j-vlm-models/smoldocling-decoder.sdz");

        // Load optimized SDZ
        assertTrue(optSdz.exists(), "Optimized SDZ must exist at " + optSdz);
        log.info("Loading optimized SDZ: {} ({}MB)", optSdz, optSdz.length() / (1024 * 1024));
        SameDiff optimized = SameDiff.load(optSdz, false);
        log.info("Loaded optimized graph: {} ops, {} vars", optimized.getOps().size(), optimized.getVariables().size());

        // Op type distribution
        Map<String, Integer> opCounts = countOpsByType(optimized);
        log.info("");
        log.info("========== OPTIMIZED SDZ OP DISTRIBUTION ==========");
        opCounts.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .forEach(e -> log.info("  {} x {}", String.format("%6d", e.getValue()), e.getKey()));

        // Focus on normalization ops
        log.info("");
        log.info("========== NORMALIZATION OPS ==========");
        for (SameDiffOp op : optimized.getOps().values()) {
            String opName = op.getOp().opName();
            if (opName.contains("rms") || opName.contains("norm") || opName.contains("layer_norm")) {
                log.info("  op={} name={}", opName, op.getName());
                log.info("    inputs: {}", op.getInputsToOp());
                log.info("    outputs: {}", op.getOutputsOfOp());
            }
        }

        // Focus on skip_rms_norm ops — trace layer 0 fully
        log.info("");
        log.info("========== LAYER 0 skip_rms_norm TRACE ==========");
        for (SameDiffOp op : optimized.getOps().values()) {
            String opName = op.getOp().opName();
            if (opName.equals("skip_rms_norm") && op.getName().contains("layers.0")) {
                log.info("OP: {} ({})", op.getName(), opName);
                log.info("  inputs: {}", op.getInputsToOp());
                log.info("  outputs: {}", op.getOutputsOfOp());
                // Trace each input
                if (op.getInputsToOp() != null) {
                    for (String inputName : op.getInputsToOp()) {
                        SDVariable v = optimized.getVariable(inputName);
                        var vi = optimized.getVariables().get(inputName);
                        log.info("  INPUT '{}': type={}, dtype={}, producedBy={}",
                                inputName,
                                v != null ? v.getVariableType() : "?",
                                v != null ? v.dataType() : "?",
                                vi != null ? vi.getOutputOfOp() : "none");
                    }
                }
                // Trace each output
                if (op.getOutputsOfOp() != null) {
                    for (String outputName : op.getOutputsOfOp()) {
                        SDVariable v = optimized.getVariable(outputName);
                        var vi = optimized.getVariables().get(outputName);
                        List<String> consumers = vi != null ? vi.getInputsForOp() : null;
                        log.info("  OUTPUT '{}': type={}, dtype={}, consumedBy={}",
                                outputName,
                                v != null ? v.getVariableType() : "?",
                                v != null ? v.dataType() : "?",
                                consumers);
                    }
                }
            }
        }

        // Focus on attention ops
        log.info("");
        log.info("========== ATTENTION OPS (layer 0) ==========");
        for (SameDiffOp op : optimized.getOps().values()) {
            String opName = op.getOp().opName();
            if ((opName.contains("attention") || opName.contains("mha") || opName.contains("dot_product"))
                    && op.getName().contains("layers.0")) {
                log.info("OP: {} ({})", op.getName(), opName);
                log.info("  inputs: {}", op.getInputsToOp());
                log.info("  outputs: {}", op.getOutputsOfOp());
            }
        }

        // Count skip_rms_norm output counts (how many outputs per op)
        log.info("");
        log.info("========== skip_rms_norm OUTPUT COUNTS ==========");
        int skipRmsTotal = 0;
        Map<Integer, Integer> outputCountDist = new TreeMap<>();
        for (SameDiffOp op : optimized.getOps().values()) {
            if (op.getOp().opName().equals("skip_rms_norm")) {
                skipRmsTotal++;
                int numOut = op.getOutputsOfOp() != null ? op.getOutputsOfOp().size() : 0;
                outputCountDist.merge(numOut, 1, Integer::sum);
            }
        }
        log.info("Total skip_rms_norm ops: {}", skipRmsTotal);
        log.info("Output count distribution: {}", outputCountDist);

        // Also load and compare raw SDZ
        if (rawSdz.exists()) {
            log.info("");
            log.info("========== RAW (unoptimized) SDZ ==========");
            SameDiff raw = SameDiff.load(rawSdz, false);
            Map<String, Integer> rawOpCounts = countOpsByType(raw);
            log.info("Raw graph: {} ops, {} vars", raw.getOps().size(), raw.getVariables().size());
            log.info("Op distribution:");
            rawOpCounts.entrySet().stream()
                    .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                    .limit(30)
                    .forEach(e -> log.info("  {} x {}", String.format("%6d", e.getValue()), e.getKey()));

            // Show raw normalization ops for comparison
            log.info("");
            log.info("Raw norm ops (layer 0):");
            for (SameDiffOp op : raw.getOps().values()) {
                String opn = op.getOp().opName();
                if ((opn.contains("rms") || opn.contains("norm")) && op.getName().contains("layers.0")) {
                    log.info("  {} ({}) inputs={} outputs={}", op.getName(), opn, op.getInputsToOp(), op.getOutputsOfOp());
                }
            }
        }

        // Check: do output_3 variables exist?
        log.info("");
        log.info("========== output_3 VARIABLE CHECK ==========");
        int found3 = 0, missing3 = 0;
        for (SameDiffOp op : optimized.getOps().values()) {
            if (op.getOp().opName().equals("skip_rms_norm")) {
                String base = op.getName().replace("/output_0", "");
                String out3Name = base + "/output_3";
                SDVariable out3Var = optimized.getVariable(out3Name);
                if (out3Var != null) {
                    var vi3 = optimized.getVariables().get(out3Name);
                    log.info("  FOUND {}: type={}, producedBy={}, consumedBy={}", out3Name,
                            out3Var.getVariableType(), vi3 != null ? vi3.getOutputOfOp() : "?",
                            vi3 != null ? vi3.getInputsForOp() : "?");
                    found3++;
                } else {
                    // Check if any variable references this as input
                    boolean referenced = false;
                    for (SameDiffOp otherOp : optimized.getOps().values()) {
                        if (otherOp.getInputsToOp() != null && otherOp.getInputsToOp().contains(out3Name)) {
                            log.info("  MISSING {} but REFERENCED by {}", out3Name, otherOp.getName());
                            referenced = true;
                        }
                    }
                    if (!referenced) {
                        missing3++;
                    } else {
                        missing3++;
                    }
                }
            }
        }
        log.info("output_3 found: {}, missing: {}", found3, missing3);

        // Also check raw SDZ for output_3
        if (rawSdz.exists()) {
            SameDiff rawForCheck = SameDiff.load(rawSdz, false);
            log.info("");
            log.info("========== RAW SDZ output_3 CHECK ==========");
            int rawFound3 = 0, rawMissing3 = 0;
            for (SameDiffOp op : rawForCheck.getOps().values()) {
                if (op.getOp().opName().equals("skip_rms_norm")) {
                    String base = op.getName().replace("/output_0", "");
                    String out3Name = base + "/output_3";
                    SDVariable out3Var = rawForCheck.getVariable(out3Name);
                    if (out3Var != null) {
                        var vi3 = rawForCheck.getVariables().get(out3Name);
                        log.info("  FOUND {}: type={}, producedBy={}, consumedBy={}", out3Name,
                                out3Var.getVariableType(), vi3 != null ? vi3.getOutputOfOp() : "?",
                                vi3 != null ? vi3.getInputsForOp() : "?");
                        rawFound3++;
                    } else {
                        rawMissing3++;
                    }
                }
            }
            log.info("RAW output_3 found: {}, missing: {}", rawFound3, rawMissing3);

            // Check how many outputs each skip_rms_norm has in raw SDZ
            Map<Integer, Integer> rawOutputCountDist = new TreeMap<>();
            for (SameDiffOp op : rawForCheck.getOps().values()) {
                if (op.getOp().opName().equals("skip_rms_norm")) {
                    int numOut = op.getOutputsOfOp() != null ? op.getOutputsOfOp().size() : 0;
                    rawOutputCountDist.merge(numOut, 1, Integer::sum);
                }
            }
            log.info("RAW skip_rms_norm output count distribution: {}", rawOutputCountDist);
        }

        // Outputs
        log.info("");
        log.info("========== GRAPH OUTPUTS ==========");
        log.info("Outputs: {}", optimized.outputs());

        // Placeholders
        log.info("");
        log.info("========== PLACEHOLDERS ==========");
        for (SDVariable v : optimized.variables()) {
            if (v.getVariableType() == VariableType.PLACEHOLDER) {
                log.info("  {} dtype={} shape={}", v.name(), v.dataType(),
                        v.getShape() != null ? Arrays.toString(v.getShape()) : "dynamic");
            }
        }

        // Check attn_mask_reformat variables
        log.info("");
        log.info("========== attn_mask_reformat VARIABLES ==========");
        for (SDVariable v : optimized.variables()) {
            if (v.name().contains("attn_mask_reformat")) {
                var vi = optimized.getVariables().get(v.name());
                log.info("  {} type={} dtype={} producedBy={} consumedBy={}",
                        v.name(), v.getVariableType(), v.dataType(),
                        vi != null ? vi.getOutputOfOp() : "?",
                        vi != null ? vi.getInputsForOp() : "?");
            }
        }

        // Check ALL VARIABLE-typed nodes that have producers (computed intermediates)
        log.info("");
        log.info("========== COMPUTED VARIABLES (type=VARIABLE but has producer) ==========");
        int computedVarCount = 0;
        for (SDVariable v : optimized.variables()) {
            if (v.getVariableType() == VariableType.VARIABLE) {
                var vi = optimized.getVariables().get(v.name());
                String producer = vi != null ? vi.getOutputOfOp() : null;
                if (producer != null && !producer.isEmpty() && optimized.getOps().containsKey(producer)) {
                    log.info("  {} producedBy={}", v.name(), producer);
                    computedVarCount++;
                }
            }
        }
        log.info("Total computed VARIABLE nodes: {}", computedVarCount);
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    private void reportOptimization(SameDiff original, String modelName) {
        List<String> outputs = original.outputs() != null ? new ArrayList<>(original.outputs()) : new ArrayList<>();
        if (outputs.isEmpty()) {
            log.warn("No outputs defined for {}, skipping", modelName);
            return;
        }

        int totalOpsBefore = original.getOps().size();
        Map<String, Integer> opsBefore = countOpsByType(original);

        log.info("BEFORE: {} ops, {} variables", totalOpsBefore, original.getVariables().size());
        log.info("Op distribution (top 15):");
        printOpDistribution(opsBefore, 15);

        long optStart = System.currentTimeMillis();
        SameDiff optimized = GraphOptimizer.optimize(original, outputs);
        long optTime = System.currentTimeMillis() - optStart;

        int totalOpsAfter = optimized.getOps().size();
        Map<String, Integer> opsAfter = countOpsByType(optimized);

        log.info("");
        log.info("AFTER: {} ops, {} variables ({}ms)", totalOpsAfter, optimized.getVariables().size(), optTime);
        log.info("Op distribution (top 15):");
        printOpDistribution(opsAfter, 15);

        // Changes
        log.info("");
        log.info("CHANGES:");
        Set<String> allOpTypes = new TreeSet<>();
        allOpTypes.addAll(opsBefore.keySet());
        allOpTypes.addAll(opsAfter.keySet());
        for (String opType : allOpTypes) {
            int before = opsBefore.getOrDefault(opType, 0);
            int after = opsAfter.getOrDefault(opType, 0);
            if (before != after) {
                log.info("  {}: {} -> {}", opType, before, after);
            }
        }

        double pct = totalOpsBefore > 0 ? (100.0 * (totalOpsBefore - totalOpsAfter) / totalOpsBefore) : 0;
        log.info("");
        log.info("SUMMARY: {} -> {} ops ({}% reduction, {}ms)", totalOpsBefore, totalOpsAfter,
                String.format("%.1f", pct), optTime);
    }

    private static Map<String, Integer> countOpsByType(SameDiff sd) {
        Map<String, Integer> counts = new TreeMap<>();
        for (SameDiffOp op : sd.getOps().values()) {
            DifferentialFunction fn = op.getOp();
            String typeName = fn != null ? fn.opName() : "null";
            counts.merge(typeName, 1, Integer::sum);
        }
        return counts;
    }

    private static Map<VariableType, Integer> countVarsByType(SameDiff sd) {
        Map<VariableType, Integer> counts = new EnumMap<>(VariableType.class);
        for (SDVariable v : sd.variables()) {
            counts.merge(v.getVariableType(), 1, Integer::sum);
        }
        return counts;
    }

    private static Map<DataType, Integer> countConstantsByDtype(SameDiff sd) {
        Map<DataType, Integer> counts = new TreeMap<>();
        for (SDVariable v : sd.variables()) {
            if (v.getVariableType() == VariableType.CONSTANT) {
                DataType dt = v.dataType();
                if (dt != null) {
                    counts.merge(dt, 1, Integer::sum);
                }
            }
        }
        return counts;
    }

    private static void printOpDistribution(Map<String, Integer> counts, int topN) {
        counts.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(topN)
                .forEach(e -> log.info("  {} x {}", String.format("%6d", e.getValue()), e.getKey()));
    }
}
