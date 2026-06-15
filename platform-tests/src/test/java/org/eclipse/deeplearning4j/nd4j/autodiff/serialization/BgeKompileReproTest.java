/*
 * ******************************************************************************
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

package org.eclipse.deeplearning4j.nd4j.autodiff.serialization;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Reproduces the exact concat_9 failure seen when kompile-app-main's
 * embedding subprocess loads bge-base-en-v1.5 from the local registry
 * at ~/.kompile/models/bge-base-en-v1.5/model.sdz.
 *
 * The subprocess feeds:
 *   input_ids -> shape=[1, 512], dtype=LONG
 *   first tokens: [101, 2023, 2003, 1037, 27354, ...] (CLS + "this is a validation...")
 *   output requested: "last_hidden_state"
 *
 * Error observed:
 *   InferenceSession -- Failed to execute operation: concat_9
 *   GenericDenseSameDiffEncoder -- Error during GenericDense encoding
 *   Model returned null embedding - inference failed silently
 *
 * This test uses NO Kompile or Anserini abstractions — pure SameDiff only.
 */
@Slf4j
public class BgeKompileReproTest extends BaseND4JTest {

    /**
     * Path to the exact model file used by the kompile subprocess.
     * This is the model that the staging service writes to the local registry.
     */
    private static final String KOMPILE_MODEL_PATH =
            System.getProperty("user.home") + "/.kompile/models/bge-base-en-v1.5/model.sdz";

    /**
     * Path to the model used by the existing BgeModelLoadingTest (different file).
     * Included for comparison if the kompile model is not available.
     */
    private static final String PLATFORM_TESTS_MODEL_PATH =
            System.getProperty("bge.model.path",
                    System.getProperty("user.dir") + "/bge-base-en-v1.5.sdz");

    @Override
    public long getTimeoutMilliseconds() {
        return 5 * 60 * 1000L;
    }

    /**
     * Exact reproduction of the kompile subprocess failure.
     *
     * The subprocess does:
     * 1. Load model via SDZSerializer
     * 2. Discover inputs = [input_ids], outputs = [last_hidden_state, 1492]
     * 3. Create input_ids with shape [1, 512], fill with tokenized
     *    "This is a validation test for the embedding model."
     * 4. Call model.output() requesting "last_hidden_state"
     * 5. concat_9 fails
     */
    @Test
    public void testKompileSubprocessReproduction() throws Exception {
        File modelFile = new File(KOMPILE_MODEL_PATH);
        assumeTrue(modelFile.exists(),
                "Kompile model not found at: " + KOMPILE_MODEL_PATH +
                ". Stage the model via kompile-model-staging first.");

        log.info("=== KOMPILE SUBPROCESS REPRODUCTION TEST ===");
        log.info("Loading model from: {}", modelFile.getAbsolutePath());
        log.info("File size: {} MB", modelFile.length() / (1024 * 1024));

        // Step 1: Load exactly as the subprocess does
        SameDiff model = SDZSerializer.load(modelFile, true);
        assertNotNull(model);

        List<String> inputs = model.inputs();
        List<String> outputs = model.outputs();
        log.info("Model inputs: {}", inputs);
        log.info("Model outputs: {}", outputs);

        // Log what the subprocess sees
        log.info("Variable count: {}", model.variables().size());
        log.info("Op count: {}", model.ops().length);

        // Step 2: The subprocess provides ONLY input_ids (no attention_mask, no token_type_ids)
        // This is the exact input the GenericDenseSameDiffEncoder creates.
        // Token IDs for "This is a validation test for the embedding model."
        // Tokenized by WordPiece with vocab.txt: [CLS]=101, this=2023, is=2003, a=1037,
        // valid=27354, ##ation=10057, test=3231, for=2005, the=1996, em=7861,
        // ##bed=8270, ##ding=4667, model=2944, .=1012, [SEP]=102, rest=0 (padding)
        long[] tokenIds = new long[512];
        tokenIds[0] = 101;   // [CLS]
        tokenIds[1] = 2023;  // this
        tokenIds[2] = 2003;  // is
        tokenIds[3] = 1037;  // a
        tokenIds[4] = 27354; // valid
        tokenIds[5] = 10057; // ##ation
        tokenIds[6] = 3231;  // test
        tokenIds[7] = 2005;  // for
        tokenIds[8] = 1996;  // the
        tokenIds[9] = 7861;  // em
        tokenIds[10] = 8270; // ##bed
        tokenIds[11] = 4667; // ##ding
        tokenIds[12] = 2944; // model
        tokenIds[13] = 1012; // .
        tokenIds[14] = 102;  // [SEP]
        // positions 15-511 remain 0 (padding)

        INDArray inputIds = Nd4j.createFromArray(tokenIds).reshape(1, 512).castTo(DataType.INT64);
        log.info("input_ids shape={}, dtype={}, first5={}",
                Arrays.toString(inputIds.shape()), inputIds.dataType(),
                Arrays.toString(new long[]{
                        inputIds.getLong(0, 0), inputIds.getLong(0, 1),
                        inputIds.getLong(0, 2), inputIds.getLong(0, 3),
                        inputIds.getLong(0, 4)}));

        // Step 3: Build placeholder map — ONLY input_ids, matching subprocess behavior
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("input_ids", inputIds);

        // Step 4: Run inference requesting "last_hidden_state" (first output)
        // This is where concat_9 fails in the subprocess
        log.info("Running inference with only input_ids (no attention_mask)...");
        try {
            Map<String, INDArray> result = model.output(placeholders, List.of("last_hidden_state"));

            INDArray output = result.get("last_hidden_state");
            assertNotNull(output, "last_hidden_state should not be null");
            log.info("last_hidden_state shape={}, dtype={}", Arrays.toString(output.shape()), output.dataType());

            // Print first few values
            INDArray flat = output.reshape(-1);
            StringBuilder sb = new StringBuilder("First 10 values: ");
            for (int i = 0; i < Math.min(10, flat.length()); i++) {
                sb.append(flat.getFloat(i)).append(", ");
            }
            log.info(sb.toString());

            // Validate
            assertFalse(output.isNaN().any(), "Output contains NaN");
            assertFalse(output.isInfinite().any(), "Output contains Inf");
            log.info("PASSED: Inference succeeded with only input_ids");

        } catch (Exception e) {
            log.error("FAILED: Inference failed (this reproduces the subprocess error)", e);
            fail("Inference failed with: " + e.getMessage());
        } finally {
            inputIds.close();
        }
    }

    /**
     * Same test but with attention_mask added — to verify if providing
     * attention_mask fixes the concat_9 failure.
     */
    @Test
    public void testWithAttentionMask() throws Exception {
        File modelFile = new File(KOMPILE_MODEL_PATH);
        assumeTrue(modelFile.exists(),
                "Kompile model not found at: " + KOMPILE_MODEL_PATH);

        log.info("=== TEST WITH ATTENTION MASK ===");
        SameDiff model = SDZSerializer.load(modelFile, true);
        assertNotNull(model);

        List<String> inputs = model.inputs();
        log.info("Model inputs: {}", inputs);

        // Check if model actually declares attention_mask as an input
        boolean hasAttentionMask = inputs.stream()
                .anyMatch(n -> n.toLowerCase().contains("attention") || n.toLowerCase().contains("mask"));
        log.info("Model declares attention_mask input: {}", hasAttentionMask);

        // Same token IDs as above
        long[] tokenIds = new long[512];
        tokenIds[0] = 101;
        tokenIds[1] = 2023;
        tokenIds[2] = 2003;
        tokenIds[3] = 1037;
        tokenIds[4] = 27354;
        tokenIds[5] = 10057;
        tokenIds[6] = 3231;
        tokenIds[7] = 2005;
        tokenIds[8] = 1996;
        tokenIds[9] = 7861;
        tokenIds[10] = 8270;
        tokenIds[11] = 4667;
        tokenIds[12] = 2944;
        tokenIds[13] = 1012;
        tokenIds[14] = 102;

        INDArray inputIds = Nd4j.createFromArray(tokenIds).reshape(1, 512).castTo(DataType.INT64);

        // Create attention_mask: 1 for real tokens (0-14), 0 for padding (15-511)
        long[] maskValues = new long[512];
        for (int i = 0; i <= 14; i++) {
            maskValues[i] = 1;
        }
        INDArray attentionMask = Nd4j.createFromArray(maskValues).reshape(1, 512).castTo(DataType.INT64);

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("input_ids", inputIds);

        // Provide attention_mask to ALL input names that look like attention/mask
        for (String inputName : inputs) {
            if (inputName.toLowerCase().contains("attention") || inputName.toLowerCase().contains("mask")) {
                placeholders.put(inputName, attentionMask);
                log.info("Added attention_mask as '{}'", inputName);
            }
        }

        // If model doesn't declare attention_mask but it might be expected
        // as a graph variable (not placeholder), log that
        if (!hasAttentionMask) {
            log.info("Model has no attention_mask placeholder — checking if it exists as a variable...");
            boolean hasMaskVar = model.variables().stream()
                    .anyMatch(v -> v.name().toLowerCase().contains("attention_mask"));
            log.info("attention_mask variable exists: {}", hasMaskVar);
        }

        log.info("Running inference with {} placeholders: {}", placeholders.size(), placeholders.keySet());

        try {
            Map<String, INDArray> result = model.output(placeholders, List.of("last_hidden_state"));

            INDArray output = result.get("last_hidden_state");
            assertNotNull(output, "last_hidden_state should not be null");
            log.info("last_hidden_state shape={}", Arrays.toString(output.shape()));

            INDArray flat = output.reshape(-1);
            StringBuilder sb = new StringBuilder("First 10 values: ");
            for (int i = 0; i < Math.min(10, flat.length()); i++) {
                sb.append(flat.getFloat(i)).append(", ");
            }
            log.info(sb.toString());

            assertFalse(output.isNaN().any(), "Output contains NaN");
            assertFalse(output.isInfinite().any(), "Output contains Inf");
            log.info("PASSED: Inference succeeded with attention_mask");

        } catch (Exception e) {
            log.error("FAILED: Inference still fails with attention_mask", e);
            fail("Inference failed even with attention_mask: " + e.getMessage());
        } finally {
            inputIds.close();
            attentionMask.close();
        }
    }

    /**
     * Diagnostic test: inspect the concat_9 operation to understand what
     * inputs it expects and why it fails.
     */
    @Test
    public void testDiagnoseConcat9() throws Exception {
        File modelFile = new File(KOMPILE_MODEL_PATH);
        assumeTrue(modelFile.exists(),
                "Kompile model not found at: " + KOMPILE_MODEL_PATH);

        log.info("=== DIAGNOSING concat_9 OPERATION ===");
        SameDiff model = SDZSerializer.load(modelFile, true);
        assertNotNull(model);

        // Find concat_9 and all concat ops
        var ops = model.getOps();
        log.info("Total ops: {}", ops.size());

        int concatCount = 0;
        for (var entry : ops.entrySet()) {
            String opName = entry.getKey();
            var opDef = entry.getValue();
            String actualOpName = opDef.getOp() != null ? opDef.getOp().opName() : "null";

            if (opName.contains("concat") || "concat".equals(actualOpName)) {
                concatCount++;
                log.info("CONCAT OP: name='{}', opName='{}', inputs={}, outputs={}",
                        opName, actualOpName,
                        opDef.getInputsToOp(), opDef.getOutputsOfOp());

                // For concat_9 specifically, dig deeper
                if (opName.equals("concat_9")) {
                    log.info("  >>> THIS IS THE FAILING OP <<<");

                    // Check each input to concat_9
                    if (opDef.getInputsToOp() != null) {
                        for (String inputVar : opDef.getInputsToOp()) {
                            SDVariable var = model.getVariable(inputVar);
                            if (var != null) {
                                log.info("  Input '{}': type={}, shape={}, dtype={}",
                                        inputVar,
                                        var.getVariableType(),
                                        var.getShape() != null ? Arrays.toString(var.getShape()) : "dynamic",
                                        var.dataType());

                                // Check what op produces this input
                                String producingOp = model.getVariables().get(inputVar).getOutputOfOp();
                                if (producingOp != null) {
                                    var prodOpDef = ops.get(producingOp);
                                    if (prodOpDef != null) {
                                        log.info("    produced by: '{}' (opName='{}')",
                                                producingOp,
                                                prodOpDef.getOp() != null ? prodOpDef.getOp().opName() : "null");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        log.info("Total concat ops: {}", concatCount);

        // Also list all model inputs and check if attention_mask is expected
        log.info("\nModel declared inputs: {}", model.inputs());
        log.info("Model declared outputs: {}", model.outputs());

        // Check for any variables with "mask" or "attention" in the name
        log.info("\nVariables containing 'mask' or 'attention':");
        for (var v : model.variables()) {
            if (v.name().toLowerCase().contains("mask") || v.name().toLowerCase().contains("attention")) {
                log.info("  '{}': type={}, shape={}, dtype={}",
                        v.name(), v.getVariableType(),
                        v.getShape() != null ? Arrays.toString(v.getShape()) : "dynamic",
                        v.dataType());
            }
        }

        // Check for variables with "position" in the name (position embeddings)
        log.info("\nVariables containing 'position':");
        for (var v : model.variables()) {
            if (v.name().toLowerCase().contains("position")) {
                log.info("  '{}': type={}, shape={}", v.name(), v.getVariableType(),
                        v.getShape() != null ? Arrays.toString(v.getShape()) : "dynamic");
            }
        }
    }

    /**
     * Compare the kompile model vs the platform-tests model to find structural differences.
     */
    @Test
    public void testCompareModels() throws Exception {
        File kompileModel = new File(KOMPILE_MODEL_PATH);
        File ptModel = new File(PLATFORM_TESTS_MODEL_PATH);
        assumeTrue(kompileModel.exists(), "Kompile model not found");
        assumeTrue(ptModel.exists(), "Platform tests model not found");

        log.info("=== COMPARING MODELS ===");

        SameDiff m1 = SDZSerializer.load(kompileModel, true);
        SameDiff m2 = SDZSerializer.load(ptModel, true);

        log.info("Kompile model: inputs={}, outputs={}, vars={}, ops={}",
                m1.inputs(), m1.outputs(), m1.variables().size(), m1.ops().length);
        log.info("Platform model: inputs={}, outputs={}, vars={}, ops={}",
                m2.inputs(), m2.outputs(), m2.variables().size(), m2.ops().length);

        // Compare file sizes
        log.info("Kompile model size: {} MB", kompileModel.length() / (1024 * 1024));
        log.info("Platform model size: {} MB", ptModel.length() / (1024 * 1024));

        // Check if they're the same model or different
        if (m1.variables().size() != m2.variables().size()) {
            log.warn("Different variable counts: {} vs {}", m1.variables().size(), m2.variables().size());
        }
        if (m1.ops().length != m2.ops().length) {
            log.warn("Different op counts: {} vs {}", m1.ops().length, m2.ops().length);
        }

        // Check input differences
        if (!m1.inputs().equals(m2.inputs())) {
            log.warn("DIFFERENT INPUTS:");
            log.warn("  Kompile: {}", m1.inputs());
            log.warn("  Platform: {}", m2.inputs());

            // Find inputs in platform model that are missing from kompile model
            for (String input : m2.inputs()) {
                if (!m1.inputs().contains(input)) {
                    log.warn("  Missing from kompile model: '{}'", input);
                }
            }
        }

        // If platform model has more inputs (like attention_mask), that's likely the issue
        if (m2.inputs().size() > m1.inputs().size()) {
            log.warn("Platform model has MORE inputs than Kompile model.");
            log.warn("The Kompile model may have been exported/converted without attention_mask.");
            log.warn("This could cause concat operations to fail due to missing shape information.");
        }
    }
}
