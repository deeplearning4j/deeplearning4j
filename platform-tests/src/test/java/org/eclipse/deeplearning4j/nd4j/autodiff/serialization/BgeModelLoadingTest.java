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
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Arrays;
import java.util.ArrayList;
import org.bytedeco.javacpp.FloatPointer;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Minimal throwaway test for loading the BGE encoder model.
 * Uses hardcoded paths - adjust as needed for your environment.
 *
 * This test validates that the bge-base-en-v1.5.sdz model can be loaded
 * via SDZSerializer without using any anserini APIs.
 */
@Slf4j
public class BgeModelLoadingTest extends BaseND4JTest {

    // Hardcoded paths - adjust these for your environment
    private static final String MODEL_PATH = "/home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests/bge-base-en-v1.5.sdz";

    // ONNX model path for fresh import
    private static final String ONNX_MODEL_PATH = "/home/agibsonccc/Documents/GitHub/kompile/anserini-models/bge-base-en-v1.5/bge-base-en-v1.5.onnx";

    private static final String VOCAB_PATH = System.getProperty("user.home")
            + "/.kompile/models/anserini/encoders/bge-base-en-v1.5/vocab.txt";

    @Override
    public long getTimeoutMilliseconds() {
        return 5 * 60 * 1000L; // 5 minutes timeout for model loading
    }

    @BeforeEach
    public void before() {
        Nd4j.getEnvironment().setVerbose(true);
        Nd4j.getEnvironment().setDebug(true);
    }
    @Test
    public void testLoadBgeModel() throws Exception {
        File modelFile = new File(MODEL_PATH);

        // Skip test if model file doesn't exist
        assumeTrue(modelFile.exists(),
                "BGE model file not found at: " + MODEL_PATH +
                ". Download the model first or adjust the path.");

        log.info("Loading BGE model from: {}", modelFile.getAbsolutePath());
        log.info("Model file size: {} MB", modelFile.length() / (1024 * 1024));

        // Load the model using SDZSerializer
        long startTime = System.currentTimeMillis();
        SameDiff sameDiffModel = SDZSerializer.load(modelFile, true);
        long loadTime = System.currentTimeMillis() - startTime;

        log.info("Model loaded in {} ms", loadTime);

        // Basic validation
        assertNotNull(sameDiffModel, "Model should not be null");

        // Log model inputs and outputs
        List<String> inputs = sameDiffModel.inputs();
        List<String> outputs = sameDiffModel.outputs();

        log.info("Model inputs: {}", inputs);
        log.info("Model outputs: {}", outputs);

        assertFalse(inputs.isEmpty(), "Model should have at least one input");
        assertFalse(outputs.isEmpty(), "Model should have at least one output");

        // Log all variables for debugging
        log.info("All variables in model:");
        for (var sdVar : sameDiffModel.variables()) {
            log.info("  {} - shape: {}, type: {}",
                    sdVar.name(),
                    sdVar.getShape() != null ? java.util.Arrays.toString(sdVar.getShape()) : "dynamic",
                    sdVar.dataType());
        }

        // Log operations count
        log.info("Total operations in model: {}", sameDiffModel.ops().length);
    }

    @Test
    public void testBgeModelInference() throws Exception {
        File onnxFile = new File(ONNX_MODEL_PATH);

        // Skip test if ONNX file doesn't exist
        assumeTrue(onnxFile.exists(),
                "BGE ONNX model file not found at: " + ONNX_MODEL_PATH);

        log.info("============================================");
        log.info("STEP 1: Importing BGE model from ONNX...");
        log.info("============================================");
        log.info("ONNX file: {}", onnxFile.getAbsolutePath());

        // Import fresh from ONNX to use the updated import code
        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        long importStart = System.currentTimeMillis();
        SameDiff sameDiffModel = importer.runImport(ONNX_MODEL_PATH, Collections.emptyMap(), true, false);
        long importTime = System.currentTimeMillis() - importStart;
        log.info("Model imported in {} ms", importTime);

        // Get input/output names
        List<String> inputs = sameDiffModel.inputs();
        List<String> outputs = sameDiffModel.outputs();

        log.info("============================================");
        log.info("STEP 2: Checking placeholder shapes...");
        log.info("============================================");
        log.info("Inputs: {}", inputs);
        log.info("Outputs: {}", outputs);

        // Print placeholder shapes to verify they are dynamic
        for (String inputName : inputs) {
            SDVariable inputVar = sameDiffModel.getVariable(inputName);
            if (inputVar != null) {
                long[] phShape = inputVar.placeholderShape();
                log.info("Input '{}': placeholderShape={}, dtype={}",
                    inputName,
                    phShape != null ? Arrays.toString(phShape) : "null",
                    inputVar.dataType());
            }
        }

        // Create dummy input tensors for BERT-style model
        // BGE uses: input_ids, attention_mask, token_type_ids
        // Typical shapes: [batch_size, sequence_length]
        int batchSize = 1;
        int seqLength = 512; // BGE model expects full 512 sequence length

        log.info("============================================");
        log.info("STEP 3: Creating inputs with shape [{}, {}]", batchSize, seqLength);
        log.info("============================================");

        // Create placeholder inputs
        Map<String, INDArray> inputMap = new java.util.HashMap<>();

        // Find the actual input names from the model
        for (String inputName : inputs) {
            // Most BERT inputs are int64 with shape [batch, seq_len]
            INDArray input = Nd4j.zeros(org.nd4j.linalg.api.buffer.DataType.INT64, batchSize, seqLength);

            // For input_ids, put some dummy token IDs (e.g., [CLS]=101, [SEP]=102)
            if (inputName.toLowerCase().contains("input_id")) {
                input.putScalar(0, 0, 101); // [CLS] token
                input.putScalar(0, 1, 102); // [SEP] token
            } else if (inputName.toLowerCase().contains("attention") || inputName.toLowerCase().contains("mask")) {
                // Attention mask: 1 for real tokens, 0 for padding
                input.putScalar(0, 0, 1);
                input.putScalar(0, 1, 1);
            }
            // token_type_ids can stay as zeros

            inputMap.put(inputName, input);
            log.info("Created input '{}': shape={}, dtype={}", inputName, Arrays.toString(input.shape()), input.dataType());
        }

        // Run inference
        log.info("============================================");
        log.info("STEP 4: Running inference...");
        log.info("============================================");
        long startTime = System.currentTimeMillis();

        Map<String, INDArray> outputMap = sameDiffModel.output(inputMap, outputs);

        long inferenceTime = System.currentTimeMillis() - startTime;
        log.info("Inference completed in {} ms", inferenceTime);

        // Log outputs
        log.info("============================================");
        log.info("STEP 5: Checking outputs...");
        log.info("============================================");
        for (Map.Entry<String, INDArray> entry : outputMap.entrySet()) {
            INDArray output = entry.getValue();
            log.info("Output '{}': shape={}, dtype={}",
                    entry.getKey(),
                    Arrays.toString(output.shape()),
                    output.dataType());

            // For embedding models, expect shape like [batch, hidden_dim] or [batch, seq, hidden_dim]
            assertNotNull(output, "Output should not be null");
            assertTrue(output.length() > 0, "Output should have elements");

            // Check for NaN/Inf
            assertFalse(output.isNaN().any(), "Output should not contain NaN");
            assertFalse(output.isInfinite().any(), "Output should not contain Inf");
        }

        log.info("============================================");
        log.info("TEST PASSED!");
        log.info("============================================");
    }

    /**
     * Test inference on the SDZ model (not ONNX import).
     * This verifies that the serialized model can actually run inference.
     */
    @Test
    public void testBgeModelInferenceSdz() throws Exception {
        File modelFile = new File(MODEL_PATH);

        // Skip test if model file doesn't exist
        assumeTrue(modelFile.exists(),
                "BGE model file not found at: " + MODEL_PATH +
                ". Download the model first or adjust the path.");

        log.info("============================================");
        log.info("Loading BGE SDZ model for inference test...");
        log.info("============================================");

        // Load the model using SDZSerializer
        SameDiff sameDiffModel = SDZSerializer.load(modelFile, true);
        assertNotNull(sameDiffModel, "Model should not be null");

        List<String> inputs = sameDiffModel.inputs();
        List<String> outputs = sameDiffModel.outputs();
        log.info("Inputs: {}", inputs);
        log.info("Outputs: {}", outputs);

        // Create dummy inputs
        int batchSize = 1;
        int seqLength = 512; // BGE model expects 512 sequence length

        log.info("Creating inputs with shape [{}, {}]", batchSize, seqLength);

        Map<String, INDArray> inputMap = new java.util.HashMap<>();
        for (String inputName : inputs) {
            INDArray input;
            if (inputName.toLowerCase().contains("input_id")) {
                // Create realistic token IDs: [CLS] + tokens + [SEP] + padding
                input = Nd4j.zeros(org.nd4j.linalg.api.buffer.DataType.INT64, batchSize, seqLength);
                input.putScalar(0, 0, 101); // [CLS] token
                // Fill with some token IDs (using small values to simulate real tokens)
                for (int i = 1; i < 10; i++) {
                    input.putScalar(0, i, 1000 + i); // Some token IDs
                }
                input.putScalar(0, 10, 102); // [SEP] token
                // Rest stays as 0 (padding token)
            } else if (inputName.toLowerCase().contains("attention") || inputName.toLowerCase().contains("mask")) {
                // Attention mask: 1 for real tokens (positions 0-10), 0 for padding
                input = Nd4j.zeros(org.nd4j.linalg.api.buffer.DataType.INT64, batchSize, seqLength);
                for (int i = 0; i <= 10; i++) {
                    input.putScalar(0, i, 1);
                }
            } else {
                // token_type_ids: all zeros is fine for single sentence
                input = Nd4j.zeros(org.nd4j.linalg.api.buffer.DataType.INT64, batchSize, seqLength);
            }
            inputMap.put(inputName, input);
            log.info("Created input '{}': shape={}, dtype={}", inputName, Arrays.toString(input.shape()), input.dataType());
        }

        // Run inference - also request intermediate variables for debugging
        log.info("Running inference on SDZ model...");
        long startTime = System.currentTimeMillis();

        // Get intermediate variables for debugging the layer_norm issue
        List<String> outputsToFetch = new ArrayList<>(outputs);
        // Add the inputs to the layer_norm op that produces last_hidden_state
        String layerNormInput = "SkipLayerNorm_AddBias_23_summed";
        if (sameDiffModel.hasVariable(layerNormInput)) {
            outputsToFetch.add(layerNormInput);
            System.out.println("DEBUG: Will fetch intermediate variable: " + layerNormInput);
        }
        // Also check pooler input
        String poolerInput = "/pooler/dense/Gemm_output_0";
        if (sameDiffModel.hasVariable(poolerInput)) {
            outputsToFetch.add(poolerInput);
            System.out.println("DEBUG: Will fetch intermediate variable: " + poolerInput);
        }

        Map<String, INDArray> outputMap = sameDiffModel.output(inputMap, outputsToFetch);

        long inferenceTime = System.currentTimeMillis() - startTime;
        log.info("Inference completed in {} ms", inferenceTime);

        // First, let's understand the graph structure for each output
        System.out.println("============================================");
        System.out.println("INVESTIGATING GRAPH STRUCTURE FOR OUTPUTS");
        System.out.println("============================================");
        for (String outputName : outputs) {
            SDVariable outputVar = sameDiffModel.getVariable(outputName);
            if (outputVar != null) {
                System.out.println("Output '" + outputName + "': variableType=" + outputVar.getVariableType());

                // Get the op that creates this output using SameDiff ops registry
                var ops = sameDiffModel.getOps();
                String outputOfOp = sameDiffModel.getVariables().get(outputName).getOutputOfOp();
                if (outputOfOp != null) {
                    var definingOp = ops.get(outputOfOp);
                    if (definingOp != null) {
                        System.out.println("  Defining op: " + outputOfOp + " (opName=" +
                            (definingOp.getOp() != null ? definingOp.getOp().opName() : "null") + ")");
                        System.out.println("  Op inputs: " + definingOp.getInputsToOp());
                        System.out.println("  Op outputs: " + definingOp.getOutputsOfOp());
                    }
                } else {
                    System.out.println("  No defining op (possibly a placeholder or constant)");
                }
            }
        }
        System.out.println("============================================");

        // Check outputs
        for (Map.Entry<String, INDArray> entry : outputMap.entrySet()) {
            INDArray outputOrig = entry.getValue();

            // Commit to ensure all CUDA kernels have finished
            Nd4j.getExecutioner().commit();

            // === DETAILED DEBUG INFO ===
            System.out.println("\n=== DEBUG: Processing output '" + entry.getKey() + "' ===");
            System.out.println("DEBUG: Original array - isAttached=" + outputOrig.isAttached() +
                              ", isView=" + outputOrig.isView() +
                              ", offset=" + outputOrig.offset() +
                              ", shape=" + Arrays.toString(outputOrig.shape()));

            // Print buffer info
            if (outputOrig.data() != null) {
                var buffer = outputOrig.data();
                System.out.println("DEBUG: Buffer - length=" + buffer.length() +
                                  ", type=" + buffer.dataType() +
                                  ", address=" + buffer.addressPointer().address());

                // Get opaque buffer info
                var opaqueBuffer = buffer.opaqueBuffer();
                if (opaqueBuffer != null) {
                    System.out.println("DEBUG: OpaqueBuffer - primaryBuffer=" +
                                      (opaqueBuffer.primaryBuffer() != null ? opaqueBuffer.primaryBuffer().address() : "null") +
                                      ", specialBuffer=" +
                                      (opaqueBuffer.specialBuffer() != null ? opaqueBuffer.specialBuffer().address() : "null"));
                }
            }

            // Detach from workspace if attached to get a stable copy
            INDArray output = outputOrig.isAttached() ? outputOrig.detach() : outputOrig;

            if (output != outputOrig) {
                System.out.println("DEBUG: After detach - isView=" + output.isView() +
                                  ", offset=" + output.offset());
                if (output.data() != null) {
                    var buffer = output.data();
                    System.out.println("DEBUG: Detached Buffer - length=" + buffer.length() +
                                      ", address=" + buffer.addressPointer().address());
                    var opaqueBuffer = buffer.opaqueBuffer();
                    if (opaqueBuffer != null) {
                        System.out.println("DEBUG: Detached OpaqueBuffer - primaryBuffer=" +
                                          (opaqueBuffer.primaryBuffer() != null ? opaqueBuffer.primaryBuffer().address() : "null") +
                                          ", specialBuffer=" +
                                          (opaqueBuffer.specialBuffer() != null ? opaqueBuffer.specialBuffer().address() : "null"));
                    }
                }
            }

            // SIMPLIFIED TEST: Just use getFloat() without any manual sync hacks
            // If synchronization is working correctly, getFloat should trigger DEVICE→HOST sync
            System.out.println("DEBUG: === SIMPLIFIED TEST - Using getFloat() directly ===");
            Nd4j.getExecutioner().commit(); // Ensure all GPU work is done

            // Check locality first
            int locality = Nd4j.getNativeOps().dbLocality(output.data().opaqueBuffer());
            System.out.println("DEBUG: Locality after commit: " + locality + " (0=both, 1=device only, -1=host only)");

            // Force a sync to primary to see what's really on GPU
            Nd4j.getNativeOps().dbSyncToPrimary(output.data().opaqueBuffer());
            int localityAfterSync = Nd4j.getNativeOps().dbLocality(output.data().opaqueBuffer());
            System.out.println("DEBUG: Locality after dbSyncToPrimary: " + localityAfterSync);

            // Check the sum of absolute values - this will tell us if there's any data at all
            double sumAbs = output.sumNumber().doubleValue();
            System.out.println("DEBUG: Sum of values: " + sumAbs);

            // Read raw memory via FloatPointer to see what's really in HOST buffer
            System.out.print("DEBUG: Raw HOST memory (first 10 floats): ");
            if (output.data() != null && output.data().opaqueBuffer() != null) {
                var opaque = output.data().opaqueBuffer();
                if (opaque.primaryBuffer() != null) {
                    FloatPointer fp = new FloatPointer(opaque.primaryBuffer());
                    for (int i = 0; i < Math.min(10, output.length()); i++) {
                        System.out.print(fp.get(i) + " ");
                    }
                    System.out.println();
                } else {
                    System.out.println("(primaryBuffer is null!)");
                }
            }

            // Just use getFloat - it should internally call synchronizeHostData which syncs DEVICE→HOST
            INDArray flat = output.reshape(-1);
            int manualInfCount = 0;
            int manualNanCount = 0;
            System.out.print("DEBUG: First 10 values via getFloat: ");
            for (long i = 0; i < Math.min(10, flat.length()); i++) {
                float v = flat.getFloat(i);
                if (i < 10) System.out.print(v + " ");
                if (Float.isInfinite(v)) manualInfCount++;
                if (Float.isNaN(v)) manualNanCount++;
            }
            System.out.println();
            for (long i = 10; i < Math.min(1000, flat.length()); i++) {
                float v = flat.getFloat(i);
                if (Float.isInfinite(v)) manualInfCount++;
                if (Float.isNaN(v)) manualNanCount++;
            }
            System.out.println("DEBUG: Manual check (first 1k) - Inf=" + manualInfCount + ", NaN=" + manualNanCount);

            boolean hasNaN = manualNanCount > 0;
            boolean hasInf = manualInfCount > 0;

            // Get statistics
            Number minVal = output.minNumber();
            Number maxVal = output.maxNumber();
            long totalLen = output.length();

            // Log output info
            log.info("=== OUTPUT CHECK: {} ===", entry.getKey());
            log.info("  Shape: {}", Arrays.toString(output.shape()));
            log.info("  dtype: {}", output.dataType());
            log.info("  min={}, max={}", minVal, maxVal);
            log.info("  hasInf={}, hasNaN={}", hasInf, hasNaN);
            log.info("  length={}", totalLen);

            // Print first 10 values
            StringBuilder sb = new StringBuilder("  First 10 values: ");
            for (int i = 0; i < Math.min(10, flat.length()); i++) {
                sb.append(flat.getFloat(i)).append(", ");
            }
            log.info(sb.toString());

            assertNotNull(output, "Output should not be null");
            assertTrue(totalLen > 0, "Output should have elements");
            assertFalse(hasNaN, "Output should not contain NaN - got min=" + minVal + ", max=" + maxVal);
            assertFalse(hasInf, "Output should not contain Inf - got min=" + minVal + ", max=" + maxVal);
        }

        // Clean up
        for (INDArray input : inputMap.values()) {
            input.close();
        }
        for (INDArray output : outputMap.values()) {
            output.close();
        }

        log.info("SDZ inference test PASSED!");
    }

    @Test
    public void testVocabFileExists() {
        File vocabFile = new File(VOCAB_PATH);

        assumeTrue(vocabFile.exists(),
                "Vocab file not found at: " + VOCAB_PATH);

        log.info("Vocab file exists at: {}", vocabFile.getAbsolutePath());
        log.info("Vocab file size: {} KB", vocabFile.length() / 1024);

        // Try reading first few lines
        try {
            List<String> lines = Files.readAllLines(Paths.get(VOCAB_PATH));
            log.info("Vocab file has {} entries", lines.size());
            log.info("First 10 vocab entries:");
            for (int i = 0; i < Math.min(10, lines.size()); i++) {
                log.info("  {}: {}", i, lines.get(i));
            }
        } catch (Exception e) {
            log.error("Failed to read vocab file", e);
        }
    }
}
