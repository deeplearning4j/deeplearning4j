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

package org.eclipse.deeplearning4j.nd4j.autodiff.serialization;

import lombok.extern.slf4j.Slf4j;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.serde.SameDiffSerializer;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.*;

import static org.junit.Assert.*;

/**
 * Test for SameDiff flatbuffers serialization with models larger than 2GB.
 *
 * This test creates a large model that exceeds the 2GB limit to verify
 * that the 64-bit flatbuffers support works correctly.
 *
 * Note: This test is marked with @Ignore by default as it requires significant
 * memory resources and time to run.
 */
@Slf4j
public class LargeSameDiffSerializationTest extends BaseND4JTest {



    @Override
    public long getTimeoutMilliseconds() {
        return 30 * 60 * 1000L; // 30 minutes timeout
    }

    @Test
    public void testLargeModelSerialization() throws IOException {
        // Parameters to create a model larger than 2GB
        int numLayers = 10;
        int layerSize = 10000;
        Nd4j.getEnvironment().setCudaDeviceLimit(Environment.CUDA_LIMIT_MALLOC_HEAP_SIZE,9999999999L);
        System.out.println("Current malloc heap size limit: " + Nd4j.getEnvironment().cudaMallocHeapSize());
        File tempFile = new File("large-samediff-model.bin");
        log.info("Will save model to: {}", tempFile.getAbsolutePath());
        // Create the model
        SameDiff sd = SameDiff.create();

        // Input
        long[] inputShape = new long[]{1, layerSize};
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, inputShape);
        SDVariable current = input;

        // Create a large network with many dense layers
        Random r = new Random(42);
        System.out.println("Creating large neural network...");

        // Map to store original arrays for comparison after loading
        Map<String, INDArray> originalArrays = new HashMap<>();

        for (int i = 0; i < numLayers; i++) {
            String layerName = "layer_" + i;
            System.out.println("Creating layer "+ i);

            // Create large weights
            SDVariable weights = sd.var(layerName + "_w", Nd4j.rand(DataType.FLOAT, layerSize, layerSize));
            SDVariable bias = sd.var(layerName + "_b", Nd4j.rand(DataType.FLOAT, 1, layerSize));

            // Store original arrays for later comparison
            originalArrays.put(layerName + "_w", weights.getArr().dup());
            originalArrays.put(layerName + "_b", bias.getArr().dup());

            // Dense layer
            current = sd.nn.relu(current.mmul(weights).add(bias),1.0);
        }

        // Output layer
        SDVariable outputWeights = sd.var("output_w", Nd4j.rand(DataType.FLOAT, layerSize, 10));
        SDVariable outputBias = sd.var("output_b", Nd4j.rand(DataType.FLOAT, 1, 10));
        originalArrays.put("output_w", outputWeights.getArr().dup());
        originalArrays.put("output_b", outputBias.getArr().dup());

        SDVariable output = sd.nn.softmax(current.mmul(outputWeights).add(outputBias));
        sd.setOutputs(output.name());

        // Calculate and log estimated size
        long bytesPerElement = 4; // float32
        long totalElements = 0;
        for (int i = 0; i < numLayers; i++) {
            totalElements += layerSize * layerSize; // weights
            totalElements += layerSize;             // bias
        }
        totalElements += layerSize * 10 + 10;      // output layer

        long estimatedBytes = totalElements * bytesPerElement;
        double estimatedGB = estimatedBytes / (1024.0 * 1024.0 * 1024.0);
        System.out.println("Estimated model size in GB " + estimatedGB);

        // Make sure we're creating a model that's larger than 2GB
        assertTrue("Model should be larger than 2GB", estimatedBytes > 2L * 1024 * 1024 * 1024);

        // Get a list of all variable names for later verification
        Set<String> originalVariableNames = new HashSet<>(sd.variableNames());
        Map<String, long[]> originalShapes = new HashMap<>();
        Map<String, DataType> originalDataTypes = new HashMap<>();

        // Store shape and data type information for all variables
        for (SDVariable var : sd.variables()) {
            originalShapes.put(var.name(), var.getShape());
            originalDataTypes.put(var.name(), var.dataType());
        }

        // Get and store operation information
        Set<String> originalOpNames = new HashSet<>();
        Map<String, String> opTypeMap = new HashMap<>();
        for (SameDiffOp op : sd.getOps().values()) {
            originalOpNames.add(op.getName());
            opTypeMap.put(op.getName(), op.getOp().opName());
        }

        // Save the model
        System.out.println("Saving model to disk...");
        long startTime = System.currentTimeMillis();
        SameDiffSerializer.saveAutoShard(sd, tempFile, true, Collections.emptyMap());
        long endTime = System.currentTimeMillis();

        // Check file size
        long fileSizeBytes = tempFile.length();
        double fileSizeGB = fileSizeBytes / (1024.0 * 1024.0 * 1024.0);
        System.out.println("Actual file size: " + fileSizeGB);

        // Try to load the model back
        System.out.println("Attempting to load model from disk...");
        startTime = System.currentTimeMillis();
        SameDiff loadedModel = SameDiffSerializer.loadSharded(tempFile, true);
        endTime = System.currentTimeMillis();

        // Verify the model loaded correctly - basic checks
        assertEquals("Variable count mismatch", sd.variables().size(), loadedModel.variables().size());
        assertEquals("Op count mismatch", sd.ops().length, loadedModel.ops().length);

        // Verify all variable names were preserved
        Set<String> loadedVariableNames = new HashSet<>(loadedModel.variableNames());
        assertEquals("Variable names mismatch", originalVariableNames, loadedVariableNames);

        // Verify all operation names were preserved
        Set<String> loadedOpNames = new HashSet<>();
        for (SameDiffOp op : loadedModel.getOps().values()) {
            loadedOpNames.add(op.getName());
            String originalOpType = opTypeMap.get(op.getName());
            assertEquals("Op type mismatch for " + op.getName(), originalOpType, op.getOp().opName());
        }
        assertEquals("Operation names mismatch", originalOpNames, loadedOpNames);

        // Verify variable shapes and data types
        for (SDVariable var : loadedModel.variables()) {
            String name = var.name();
            assertArrayEquals("Shape mismatch for variable " + name,
                    originalShapes.get(name), var.getShape());
            assertEquals("Data type mismatch for variable " + name,
                    originalDataTypes.get(name), var.dataType());
        }

        // Verify array contents for a subset of variables (checking all would be too expensive)
        // Select one weight matrix and bias from each third of the network
        List<String> samplesToCheck = Arrays.asList(
                "layer_0_w", "layer_0_b",
                "layer_" + (numLayers/3) + "_w", "layer_" + (numLayers/3) + "_b",
                "layer_" + (2*numLayers/3) + "_w", "layer_" + (2*numLayers/3) + "_b",
                "layer_" + (numLayers-1) + "_w", "layer_" + (numLayers-1) + "_b",
                "output_w", "output_b"
        );

        for (String varName : samplesToCheck) {
            INDArray originalArray = originalArrays.get(varName);
            INDArray loadedArray = loadedModel.getVariable(varName).getArr();
            INDArray originalSdArray = sd.getVariable(varName).getArr();
            assertNotNull("Original array is null for " + varName, originalArray);
            assertNotNull("Loaded array is null for " + varName, loadedArray);
            assertArrayEquals("Array shape mismatch for " + varName,
                    originalArray.shape(), loadedArray.shape());
            assertEquals("Array data type mismatch for " + varName,
                    originalArray.dataType(), loadedArray.dataType());

            // Check if arrays are equal - use a distance metric with tolerance
            // for floating point comparisons
            boolean arraysEqual = originalArray.equalsWithEps(loadedArray,1e-5);
            assertTrue("Array contents mismatch for " + varName, arraysEqual);

            // Additionally check sum, min, max as quick indicators of array content integrity
            assertEquals("Array sum mismatch for " + varName,
                    originalArray.sumNumber().doubleValue(),
                    loadedArray.sumNumber().doubleValue(),
                    1e-3);
            assertEquals("Array min mismatch for " + varName,
                    originalArray.minNumber().doubleValue(),
                    loadedArray.minNumber().doubleValue(),
                    1e-5);
            assertEquals("Array max mismatch for " + varName,
                    originalArray.maxNumber().doubleValue(),
                    loadedArray.maxNumber().doubleValue(),
                    1e-5);
        }

        // Verify graph structure integrity by checking a few connections
        // Spot check for a couple of layers in the network
        for (int i = 0; i < numLayers; i += Math.max(1, numLayers/4)) { // Check every nth layer
            String layerWeightName = "layer_" + i + "_w";
            String layerBiasName = "layer_" + i + "_b";

            // Verify these variables exist in both original and loaded model
            assertTrue("Missing weight variable in original model: " + layerWeightName,
                    sd.hasVariable(layerWeightName));
            assertTrue("Missing bias variable in original model: " + layerBiasName,
                    sd.hasVariable(layerBiasName));
            assertTrue("Missing weight variable in loaded model: " + layerWeightName,
                    loadedModel.hasVariable(layerWeightName));
            assertTrue("Missing bias variable in loaded model: " + layerBiasName,
                    loadedModel.hasVariable(layerBiasName));
        }

        // Clean up the temp file
        boolean deleted = tempFile.delete();
        if (!deleted) {
            log.warn("Failed to delete temporary model file: {}", tempFile.getAbsolutePath());
        }

        System.out.println("Test completed successfully!");
    }


    @Test
    public void testMultipleDataTypeSerialization() throws IOException {
        // Parameters for model with multiple data types
        int numLayers = 3;
        int layerSize = 1000;

        // Configure memory
        Nd4j.getEnvironment().setCudaDeviceLimit(Environment.CUDA_LIMIT_MALLOC_HEAP_SIZE, 9999999999L);
        System.out.println("Current malloc heap size limit: " + Nd4j.getEnvironment().cudaMallocHeapSize());

        // Set up temp file
        File tempFile = new File("multi-datatype-samediff-model.bin");
        log.info("Will save model to: {}", tempFile.getAbsolutePath());

        // Create the model
        SameDiff sd = SameDiff.create();

        // Input layer - standard FLOAT
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, layerSize);
        SDVariable current = input;

        // Define data types for each layer
        DataType[] layerTypes = {
                DataType.FLOAT,  // Layer 0: Standard floating point
                DataType.DOUBLE, // Layer 1: Double precision
                DataType.HALF,   // Layer 2: Half precision
        };

        // Define memory ordering for each layer
        char[] layerOrders = {'c', 'f', 'c', 'f'};

        // Store original arrays for verification
        Map<String, INDArray> originalArrays = new HashMap<>();

        System.out.println("Creating neural network with multiple data types...");

        // Create network with layers of different data types
        for (int i = 0; i < numLayers; i++) {
            String layerName = "layer_" + i;
            DataType dtype = layerTypes[i];
            char order = layerOrders[i];

            System.out.println("Creating layer " + i + " with data type " + dtype + " and order " + order);

            // Create weights with appropriate dtype and ordering
            INDArray weightArray;
            INDArray biasArray;

            if (dtype == DataType.FLOAT || dtype == DataType.DOUBLE) {
                // For floating point types
                weightArray = Nd4j.rand(dtype, layerSize, layerSize).dup(order);
                biasArray = Nd4j.rand(dtype, 1, layerSize).dup(order);

                // Scale the values
                weightArray.muli(0.1);
                biasArray.muli(0.01);
            }
            else if (dtype == DataType.HALF) {
                // For half precision, create as float then cast
                weightArray = Nd4j.rand(DataType.FLOAT, layerSize, layerSize).dup(order).muli(0.1).castTo(DataType.HALF);
                biasArray = Nd4j.rand(DataType.FLOAT, 1, layerSize).dup(order).muli(0.01).castTo(DataType.HALF);
            }
            else {
                // For integer types, initialize with specific values
                weightArray = Nd4j.zeros(dtype, layerSize, layerSize).dup(order);
                biasArray = Nd4j.zeros(dtype, 1, layerSize).dup(order);

                // Fill with deterministic values
                for (int j = 0; j < layerSize; j++) {
                    for (int k = 0; k < layerSize; k++) {
                        weightArray.putScalar(new int[]{j, k}, (j+k) % 10);
                    }
                    biasArray.putScalar(new int[]{0, j}, j % 5);
                }
            }

            // Create variables in the graph
            SDVariable weights = sd.var(layerName + "_w", weightArray);
            SDVariable bias = sd.var(layerName + "_b", biasArray);

            // Store original arrays for later verification
            originalArrays.put(layerName + "_w", weightArray.dup());
            originalArrays.put(layerName + "_b", biasArray.dup());

            // Cast input to match current layer type if needed
            SDVariable layerInput = current;
            if (i > 0 && layerInput.dataType() != dtype) {
                layerInput = layerInput.castTo(dtype);
            }

            // Create layer operations
            SDVariable z = layerInput.mmul(weights).add(bias);

            // Handle activation - for integer types, cast to float for activation, then back
            if (dtype == DataType.INT) {
                SDVariable floatActivation = sd.nn().relu(z.castTo(DataType.FLOAT), 0.0);
                current = floatActivation.castTo(DataType.INT);
            } else {
                current = sd.nn().relu(z, 0.0);
            }
        }

        // Output layer - always use FLOAT for final output
        SDVariable outputWeights = sd.var("output_w", Nd4j.rand(DataType.FLOAT, layerSize, 10).muli(0.1));
        SDVariable outputBias = sd.var("output_b", Nd4j.rand(DataType.FLOAT, 1, 10).muli(0.01));

        originalArrays.put("output_w", outputWeights.getArr().dup());
        originalArrays.put("output_b", outputBias.getArr().dup());

        // Ensure output is float type
        if (current.dataType() != DataType.FLOAT) {
            current = current.castTo(DataType.FLOAT);
        }

        // Final output
        SDVariable output = sd.nn().softmax(current.mmul(outputWeights).add(outputBias));
        sd.setOutputs(output.name());

        // Calculate estimated size
        long estimatedBytes = 0;
        for (int i = 0; i < numLayers; i++) {
            long layerElements = (long)layerSize * layerSize + layerSize;
            estimatedBytes += layerElements * layerTypes[i].width();
        }
        estimatedBytes += ((long)layerSize * 10 + 10) * 4; // output layer in FLOAT (4 bytes)

        double estimatedGB = estimatedBytes / (1024.0 * 1024.0 * 1024.0);
        System.out.println("Estimated model size: " + estimatedGB + " GB");

        // Save the model
        System.out.println("Saving model to disk...");
        long startTime = System.currentTimeMillis();
        SameDiffSerializer.saveAutoShard(sd, tempFile, true, Collections.emptyMap());
        long endTime = System.currentTimeMillis();
        System.out.println("Save completed in " + (endTime - startTime) + " ms");

        // Verify file was created
        assertTrue("Model file or shards should exist", tempFile.exists() || isSharded(tempFile));

        // Check if file was sharded
        boolean isSharded = isSharded(tempFile);
        if (isSharded) {
            System.out.println("Model was sharded into multiple files");
        }

        // Load the model back
        System.out.println("Loading model from disk...");
        startTime = System.currentTimeMillis();
        SameDiff loadedModel;
        if (isSharded) {
            loadedModel = SameDiffSerializer.loadSharded(tempFile, true);
        } else {
            loadedModel = SameDiffSerializer.load(tempFile, true);
        }
        endTime = System.currentTimeMillis();
        System.out.println("Load completed in " + (endTime - startTime) + " ms");

        // Basic validation
        assertNotNull("Loaded model should not be null", loadedModel);
        assertEquals("Variable count mismatch", sd.variables().size(), loadedModel.variables().size());
        assertEquals("Op count mismatch", sd.ops().length, loadedModel.ops().length);

        // Verify all variable names were preserved
        Set<String> originalVariableNames = new HashSet<>(sd.variableNames());
        Set<String> loadedVariableNames = new HashSet<>(loadedModel.variableNames());
        assertEquals("Variable names mismatch", originalVariableNames, loadedVariableNames);

        // Verify all layers
        for (int i = 0; i < numLayers; i++) {
            String layerName = "layer_" + i;
            DataType expectedType = layerTypes[i];
            char expectedOrder = layerOrders[i];

            verifyVariable(loadedModel, originalArrays, layerName + "_w", expectedType, expectedOrder, layerSize, layerSize);
            verifyVariable(loadedModel, originalArrays, layerName + "_b", expectedType, expectedOrder, 1, layerSize);
        }

        // Verify output layer
        verifyVariable(loadedModel, originalArrays, "output_w", DataType.FLOAT, 'c', layerSize, 10);
        verifyVariable(loadedModel, originalArrays, "output_b", DataType.FLOAT, 'c', 1, 10);

        // Clean up
        if (isSharded) {
            deleteShardFiles(tempFile);
        } else {
            tempFile.delete();
        }

        System.out.println("Multi-datatype serialization test completed successfully");
    }

    /**
     * Helper to check if a model is stored as sharded files
     */
    private boolean isSharded(File baseFile) {
        File parentDir = baseFile.getParentFile();
        if (parentDir == null) parentDir = new File(".");
        String baseName = baseFile.getName();
        int dotIdx = baseName.lastIndexOf('.');
        if (dotIdx > 0) baseName = baseName.substring(0, dotIdx);

        String finalBaseName = baseName;
        File[] shardFiles = parentDir.listFiles((dir, name) ->
                name.startsWith(finalBaseName + ".shard") && name.endsWith(".sdnb"));

        return shardFiles != null && shardFiles.length > 0;
    }

    /**
     * Helper to delete all shard files
     */
    private void deleteShardFiles(File baseFile) {
        File parentDir = baseFile.getParentFile();
        if (parentDir == null) parentDir = new File(".");
        String baseName = baseFile.getName();
        int dotIdx = baseName.lastIndexOf('.');
        if (dotIdx > 0) baseName = baseName.substring(0, dotIdx);

        String finalBaseName = baseName;
        File[] shardFiles = parentDir.listFiles((dir, name) ->
                name.startsWith(finalBaseName + ".shard") && name.endsWith(".sdnb"));

        if (shardFiles != null) {
            for (File f : shardFiles) {
                f.delete();
            }
        }
    }

    /**
     * Helper to verify a variable's properties
     */
    private void verifyVariable(SameDiff loadedModel, Map<String, INDArray> originalArrays,
                                String varName, DataType expectedType, char expectedOrder,
                                int expectedRows, int expectedCols) {
        // Check variable exists
        assertTrue("Missing variable: " + varName, loadedModel.hasVariable(varName));

        // Get arrays
        INDArray originalArray = originalArrays.get(varName);
        INDArray loadedArray = loadedModel.getVariable(varName).getArr();

        assertNotNull("Original array should not be null for " + varName, originalArray);
        assertNotNull("Loaded array should not be null for " + varName, loadedArray);

        // Verify data type
        assertEquals("Data type mismatch for " + varName, expectedType, loadedArray.dataType());

        // Verify ordering
        assertEquals("Ordering mismatch for " + varName, expectedOrder, loadedArray.ordering());

        // Verify shape
        assertEquals("Row count mismatch for " + varName, expectedRows, loadedArray.rows());
        assertEquals("Column count mismatch for " + varName, expectedCols, loadedArray.columns());

        // Verify content at key locations
        assertEquals("First element mismatch for " + varName,
                originalArray.getDouble(0, 0),
                loadedArray.getDouble(0, 0), 1e-5);

        if (expectedRows > 1 && expectedCols > 1) {
            int midRow = expectedRows / 2;
            int midCol = expectedCols / 2;

            assertEquals("Middle element mismatch for " + varName,
                    originalArray.getDouble(midRow, midCol),
                    loadedArray.getDouble(midRow, midCol), 1e-5);
        }

        assertEquals("Last element mismatch for " + varName,
                originalArray.getDouble(expectedRows-1, expectedCols-1),
                loadedArray.getDouble(expectedRows-1, expectedCols-1), 1e-5);
    }
}