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
