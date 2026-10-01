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

package org.eclipse.deeplearning4j.nd4j.autodiff.serialization;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.autodiff.samediff.serde.SameDiffSerializer;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.graph.FlatGraph;
import org.nd4j.graph.FlatVariable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;
import java.util.zip.ZipFile;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Test for SameDiff serialization using the ZIP archive format, especially for models larger than 2GB.
 *
 * This test creates a large model that exceeds the 2GB limit to verify
 * that the ZIP-based serialization works correctly.
 *
 * Note: These tests can require significant memory resources and time to run.
 */
@Slf4j
public class LargeSameDiffSerializationTest extends BaseND4JTest {

    @Override
    public long getTimeoutMilliseconds() {
        return 30 * 60 * 1000L; // 30 minutes timeout
    }

    /**
     * Checks if enough free memory is available to likely run the test.
     * Adjust the threshold as needed.
     */
    private boolean hasEnoughMemory(long requiredBytes) {
        long maxMemory = Runtime.getRuntime().maxMemory();
        long freeMemory = Runtime.getRuntime().freeMemory();
        long totalFree = freeMemory + (maxMemory - Runtime.getRuntime().totalMemory());
        log.info("Memory Check: Max={}, Total={}, Free={}, Available={}. Required ~{}",
                maxMemory / (1024*1024), Runtime.getRuntime().totalMemory() / (1024*1024),
                freeMemory / (1024*1024), totalFree / (1024*1024), requiredBytes / (1024*1024));
        // Require slightly more available than strictly calculated for overhead
        return totalFree > requiredBytes * 1.2;
    }

    @Test
    public void testShardedSdnbSparseManifestOffsetDoesNotInflateMetadataLength() throws IOException {
        File tempDir = new File(System.getProperty("java.io.tmpdir"), "sdnb-sparse-manifest-" + UUID.randomUUID());
        assertTrue(tempDir.mkdirs(), "Could not create temporary directory: " + tempDir.getAbsolutePath());
        File baseFile = new File(tempDir, "sparse-layout.sd");

        try {
            SameDiff sd = SameDiff.create();
            int appendedElements = 300_000; // 1.2MB, above SameDiffSerializer's append threshold.
            INDArray original = Nd4j.rand(DataType.FLOAT, 1, appendedElements);
            sd.var("large", original.dup());

            SameDiffSerializer.saveSharded(sd, baseFile, false, 2, Collections.emptyMap());
            File variableShard = new File(tempDir, "sparse-layout.shard1-of-2.sdnb");
            assertTrue(variableShard.isFile(), "Variable shard was not created: " + variableShard.getAbsolutePath());

            long inflatedMetadataLength = moveManifestPast2GB(variableShard);
            assertTrue(inflatedMetadataLength > Integer.MAX_VALUE,
                    "Test setup must make manifestOffset - metadataOffset exceed Integer.MAX_VALUE");

            SameDiff loaded = SameDiffSerializer.loadSharded(baseFile, false);
            assertTrue(loaded.hasVariable("large"), "Loaded graph is missing variable");
            INDArray loadedArray = loaded.getVariable("large").getArr();
            assertNotNull(loadedArray, "Loaded appended array is null");
            assertArrayEquals(original.shape(), loadedArray.shape(), "Loaded appended array shape mismatch");
            assertEquals(original.dataType(), loadedArray.dataType(), "Loaded appended array data type mismatch");
            assertTrue(original.equalsWithEps(loadedArray, 1e-5), "Loaded appended array values changed");
        } finally {
            deleteShardFiles(baseFile);
            baseFile.delete();
            tempDir.delete();
        }
    }

    @Test
    public void testShardedAppenderWritesNativeFlatArrayDescriptorForMetadataStub() throws IOException {
        File tempDir = new File(System.getProperty("java.io.tmpdir"), "sdnb-native-descriptor-" + UUID.randomUUID());
        assertTrue(tempDir.mkdirs(), "Could not create temporary directory: " + tempDir.getAbsolutePath());
        File baseFile = new File(tempDir, "native-descriptor.sd");

        try {
            SameDiff sd = SameDiff.create();
            int appendedElements = 300_000; // 1.2MB, above the append threshold.
            INDArray original = Nd4j.rand(DataType.FLOAT, 1, appendedElements);
            sd.var("first_appended_weight", original.dup());

            SameDiffSerializer.saveSharded(sd, baseFile, false, 2, Collections.emptyMap());
            File variableShard = new File(tempDir, "native-descriptor.shard1-of-2.sdnb");
            assertTrue(variableShard.isFile(), "Variable shard was not created: " + variableShard.getAbsolutePath());

            long metadataOffset;
            try (RandomAccessFile raf = new RandomAccessFile(variableShard, "r")) {
                raf.seek(24);
                metadataOffset = raf.readLong();
            }
            byte[] fileBytes = Files.readAllBytes(variableShard.toPath());
            assertTrue(metadataOffset >= 32 && metadataOffset < fileBytes.length,
                    "Metadata offset is outside the SDNB file");

            ByteBuffer metadata = ByteBuffer.wrap(fileBytes);
            metadata.position((int) metadataOffset);
            FlatGraph flatGraph = FlatGraph.getRootAsFlatGraph(metadata);
            FlatVariable flatVariable = null;
            for (int i = 0; i < flatGraph.variablesLength(); i++) {
                FlatVariable candidate = flatGraph.variables(i);
                if ("first_appended_weight".equals(candidate.name())) {
                    flatVariable = candidate;
                    break;
                }
            }

            assertNotNull(flatVariable, "Variable metadata is missing from the variable shard");
            assertNotNull(flatVariable.ndarray(),
                    "Native FlatArray descriptor is missing for the appended tensor");
            assertEquals(0L, flatVariable.ndarray().appendedDataOffset(),
                    "The first tensor must use raw-data-relative offset zero");
            assertEquals(original.length() * original.dataType().width(),
                    flatVariable.ndarray().appendedDataLength(),
                    "Native FlatArray descriptor has the wrong byte length");
        } finally {
            deleteShardFiles(baseFile);
            baseFile.delete();
            tempDir.delete();
        }
    }

    @Test
    public void testFinalInlineShardManifestHeaderIsValid() throws IOException {
        File tempDir = new File(System.getProperty("java.io.tmpdir"), "sdnb-final-inline-manifest-" + UUID.randomUUID());
        assertTrue(tempDir.mkdirs(), "Could not create temporary directory: " + tempDir.getAbsolutePath());
        File baseFile = new File(tempDir, "inline-final.sd");

        try {
            SameDiff sd = SameDiff.create();
            INDArray smallA = Nd4j.rand(DataType.FLOAT, 4, 8);
            INDArray smallB = Nd4j.rand(DataType.FLOAT, 8, 4);
            sd.var("small_a", smallA.dup());
            sd.var("small_b", smallB.dup());

            SameDiffSerializer.saveSharded(sd, baseFile, false, 2, Collections.emptyMap());
            File finalShard = new File(tempDir, "inline-final.shard1-of-2.sdnb");
            assertTrue(finalShard.isFile(), "Final inline shard was not created: " + finalShard.getAbsolutePath());
            assertShardManifestMagic(finalShard);

            SameDiff loaded = SameDiffSerializer.loadSharded(baseFile, false);
            assertTrue(smallA.equalsWithEps(loaded.getVariable("small_a").getArr(), 1e-5),
                    "Loaded final inline shard value changed for small_a");
            assertTrue(smallB.equalsWithEps(loaded.getVariable("small_b").getArr(), 1e-5),
                    "Loaded final inline shard value changed for small_b");
        } finally {
            deleteShardFiles(baseFile);
            baseFile.delete();
            tempDir.delete();
        }
    }


    @Test
    public void testLargeModelSerialization(@TempDir Path tempDir) throws IOException {
        // Parameters to create a model larger than 2GB
        int numLayers = 10;
        int layerSize = 10000;
        Nd4j.getEnvironment().setCudaDeviceLimit(Environment.CUDA_LIMIT_MALLOC_HEAP_SIZE,9999999999L);
        System.out.println("Current malloc heap size limit: " + Nd4j.getEnvironment().cudaMallocHeapSize());
        File tempFile = tempDir.resolve("large-samediff-model.bin").toFile();
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
        assertTrue(estimatedBytes > 2L * 1024 * 1024 * 1024, "Model should be larger than 2GB");

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
        assertEquals(sd.variables().size(), loadedModel.variables().size(), "Variable count mismatch");
        assertEquals(sd.ops().length, loadedModel.ops().length, "Op count mismatch");

        // Verify all variable names were preserved
        Set<String> loadedVariableNames = new HashSet<>(loadedModel.variableNames());
        assertEquals(originalVariableNames, loadedVariableNames, "Variable names mismatch");

        // Verify all operation names were preserved
        Set<String> loadedOpNames = new HashSet<>();
        for (SameDiffOp op : loadedModel.getOps().values()) {
            loadedOpNames.add(op.getName());
            String originalOpType = opTypeMap.get(op.getName());
            assertEquals(originalOpType, op.getOp().opName(), "Op type mismatch for " + op.getName());
        }
        assertEquals(originalOpNames, loadedOpNames, "Operation names mismatch");

        // Verify variable shapes and data types
        for (SDVariable var : loadedModel.variables()) {
            String name = var.name();
            assertArrayEquals(originalShapes.get(name), var.getShape(),
                    "Shape mismatch for variable " + name);
            assertEquals(originalDataTypes.get(name), var.dataType(),
                    "Data type mismatch for variable " + name);
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
            assertNotNull(originalArray, "Original array is null for " + varName);
            assertNotNull(loadedArray, "Loaded array is null for " + varName);
            assertArrayEquals(originalArray.shape(), loadedArray.shape(),
                    "Array shape mismatch for " + varName);
            assertEquals(originalArray.dataType(), loadedArray.dataType(),
                    "Array data type mismatch for " + varName);

            // Check if arrays are equal - use a distance metric with tolerance
            // for floating point comparisons
            boolean arraysEqual = originalArray.equalsWithEps(loadedArray,1e-5);
            assertTrue(arraysEqual, "Array contents mismatch for " + varName);

            // Additionally check sum, min, max as quick indicators of array content integrity
            assertEquals(originalArray.sumNumber().doubleValue(),
                    loadedArray.sumNumber().doubleValue(),
                    1e-3,
                    "Array sum mismatch for " + varName);
            assertEquals(originalArray.minNumber().doubleValue(),
                    loadedArray.minNumber().doubleValue(),
                    1e-5,
                    "Array min mismatch for " + varName);
            assertEquals(originalArray.maxNumber().doubleValue(),
                    loadedArray.maxNumber().doubleValue(),
                    1e-5,
                    "Array max mismatch for " + varName);
        }

        // Verify graph structure integrity by checking a few connections
        // Spot check for a couple of layers in the network
        for (int i = 0; i < numLayers; i += Math.max(1, numLayers/4)) { // Check every nth layer
            String layerWeightName = "layer_" + i + "_w";
            String layerBiasName = "layer_" + i + "_b";

            // Verify these variables exist in both original and loaded model
            assertTrue(sd.hasVariable(layerWeightName),
                    "Missing weight variable in original model: " + layerWeightName);
            assertTrue(sd.hasVariable(layerBiasName),
                    "Missing bias variable in original model: " + layerBiasName);
            assertTrue(loadedModel.hasVariable(layerWeightName),
                    "Missing weight variable in loaded model: " + layerWeightName);
            assertTrue(loadedModel.hasVariable(layerBiasName),
                    "Missing bias variable in loaded model: " + layerBiasName);
        }

        System.out.println("Test completed successfully!");
    }


    /**
     * Small constants are serialized inline. One-byte float storage (FLOAT8, FLOAT8_E5M2) and
     * BFLOAT16/HALF scalars must survive dup(), the path GraphOptimizer copies graphs through:
     * a lost NVFP4 block-scale array otherwise surfaces later as a missing plan input.
     */
    @Test
    public void testInlineLowPrecisionConstantsSurviveDup() {
        SameDiff sd = SameDiff.create();
        byte[] storage = new byte[16 * 20];
        for (int i = 0; i < storage.length; i++) storage[i] = (byte) (i * 37 + 11);
        Map<String, INDArray> originals = new LinkedHashMap<>();
        originals.put("e4m3", rawBytes(DataType.FLOAT8, storage, 16, 20));
        originals.put("e5m2", rawBytes(DataType.FLOAT8_E5M2, storage, 16, 20));
        originals.put("e4m3_scalar", rawBytes(DataType.FLOAT8, new byte[]{0x3A}));
        originals.put("bf16_scalar", Nd4j.scalar(DataType.BFLOAT16, 3.140625));
        originals.put("half_scalar", Nd4j.scalar(DataType.HALF, 1.5));
        originals.forEach(sd::constant);

        SameDiff copy = sd.dup();
        for (Map.Entry<String, INDArray> entry : originals.entrySet()) {
            INDArray expected = entry.getValue();
            INDArray actual = copy.getArrForVarName(entry.getKey());
            assertNotNull(actual, "array lost in dup(): " + entry.getKey());
            assertEquals(expected.dataType(), actual.dataType(), entry.getKey());
            assertArrayEquals(expected.shape(), actual.shape(), entry.getKey());
            if (expected.dataType() == DataType.FLOAT8 || expected.dataType() == DataType.FLOAT8_E5M2) {
                assertArrayEquals(storageBytes(expected), storageBytes(actual), entry.getKey());
            } else {
                assertEquals(expected.getDouble(0), actual.getDouble(0), 0.0, entry.getKey());
            }
        }
    }

    private static INDArray rawBytes(DataType dtype, byte[] bytes, long... shape) {
        INDArray array = Nd4j.createUninitialized(dtype, shape, 'c');
        new org.bytedeco.javacpp.BytePointer(array.data().pointer()).capacity(bytes.length).put(bytes);
        Nd4j.getAffinityManager().tagLocation(array, org.nd4j.linalg.api.concurrency.AffinityManager.Location.HOST);
        return array;
    }

    private static byte[] storageBytes(INDArray array) {
        Nd4j.getAffinityManager().ensureLocation(array, org.nd4j.linalg.api.concurrency.AffinityManager.Location.HOST);
        byte[] bytes = new byte[(int) array.length()];
        new org.bytedeco.javacpp.BytePointer(array.data().pointer()).capacity(bytes.length).get(bytes);
        return bytes;
    }

    @Test
    public void testMultipleDataTypeSerialization(@TempDir Path tempDir) throws IOException {
        // Parameters for model with multiple data types
        int numLayers = 3;
        int layerSize = 1000;

        // Configure memory
        Nd4j.getEnvironment().setCudaDeviceLimit(Environment.CUDA_LIMIT_MALLOC_HEAP_SIZE, 9999999999L);
        System.out.println("Current malloc heap size limit: " + Nd4j.getEnvironment().cudaMallocHeapSize());

        // Set up temp file
        File tempFile = tempDir.resolve("multi-datatype-samediff-model.bin").toFile();
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
        assertTrue(tempFile.exists() || isSharded(tempFile), "Model file or shards should exist");

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
        assertNotNull(loadedModel, "Loaded model should not be null");
        assertEquals(sd.variables().size(), loadedModel.variables().size(), "Variable count mismatch");
        assertEquals(sd.ops().length, loadedModel.ops().length, "Op count mismatch");

        // Verify all variable names were preserved
        Set<String> originalVariableNames = new HashSet<>(sd.variableNames());
        Set<String> loadedVariableNames = new HashSet<>(loadedModel.variableNames());
        assertEquals(originalVariableNames, loadedVariableNames, "Variable names mismatch");

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

        System.out.println("Multi-datatype serialization test completed successfully");
    }


    @Test
    //@Ignore // Uncomment if test consistently fails due to resource limits
    public void testLargeModelSerializationZip(@TempDir Path tempDir) throws IOException {
        // --- This test verifies the ZIP format (.sdz) ---
        // --- It should use SDZSerializer ---

        // Parameters (same as before)
        int numLayers = 10; int layerSize = 8000; // Reduced size slightly if needed for CI/local runs
        long bytesPerElement = 4;
        long estimatedElements = ((long)layerSize * layerSize + layerSize) * numLayers + (long)layerSize * 10 + 10;
        long estimatedBytes = estimatedElements * bytesPerElement;
        double estimatedGB = estimatedBytes / (1024.0 * 1024.0 * 1024.0);
        log.info("Estimated model size: {:.2f} GB", estimatedGB);
        // Adjust memory check and assumption as needed
        assumeTrue(estimatedBytes > 1.5 * 1024 * 1024 * 1024 && hasEnoughMemory(estimatedBytes),
                "Skipping large ZIP test: Insufficient memory or model size requirement not met"); // Example: 1.5GB threshold
        Nd4j.getEnvironment().setCudaDeviceLimit(Environment.CUDA_LIMIT_MALLOC_HEAP_SIZE, estimatedBytes * 2); // Request double memory
        System.out.println("Attempting CUDA malloc heap size limit: " + Nd4j.getEnvironment().cudaMallocHeapSize());

        File targetZipFile = tempDir.resolve("large-samediff-model.sdz").toFile();
        log.info("Target final ZIP file: {}", targetZipFile.getAbsolutePath());

        // Create the model (same as before)
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, layerSize);
        SDVariable current = input;
        Map<String, INDArray> originalArrays = new HashMap<>();
        log.info("Creating large neural network...");
        for (int i = 0; i < numLayers; i++) {
            String layerName = "layer_" + i;
            log.debug("Creating layer {}", i);
            SDVariable weights = sd.var(layerName + "_w", Nd4j.rand(DataType.FLOAT, layerSize, layerSize));
            SDVariable bias = sd.var(layerName + "_b", Nd4j.rand(DataType.FLOAT, 1, layerSize));
            originalArrays.put(layerName + "_w", weights.getArr().dup());
            originalArrays.put(layerName + "_b", bias.getArr().dup());
            current = sd.nn.relu(current.mmul(weights).add(bias), 0.0); // Use 0.0 for relu slope
        }
        SDVariable outputWeights = sd.var("output_w", Nd4j.rand(DataType.FLOAT, layerSize, 10));
        SDVariable outputBias = sd.var("output_b", Nd4j.rand(DataType.FLOAT, 1, 10));
        originalArrays.put("output_w", outputWeights.getArr().dup());
        originalArrays.put("output_b", outputBias.getArr().dup());
        SDVariable output = sd.nn.softmax(current.mmul(outputWeights).add(outputBias));
        sd.setOutputs(output.name());
        log.info("Model creation finished.");

        // --- Pre-save checks ---
        Set<String> originalVariableNames = new HashSet<>(sd.variableNames());
        Map<String, long[]> originalShapes = sd.variables().stream()
                .filter(v -> v.name() != null && v.getShape() != null)
                .collect(Collectors.toMap(SDVariable::name, SDVariable::getShape));
        Map<String, DataType> originalDataTypes = sd.variables().stream()
                .filter(v -> v.name() != null && v.dataType() != null) // Ensure name and dtype are not null
                .collect(Collectors.toMap(SDVariable::name, SDVariable::dataType, (dt1, dt2) -> {
                    log.warn("Duplicate variable name found during pre-save check: {}. Using first encountered datatype: {}", dt1);
                    return dt1; // Handle potential duplicates if any
                }));
        Set<String> originalOpNames = sd.getOps().values().stream().map(SameDiffOp::getName).collect(Collectors.toSet());


        // --- Save using the correct SDZSerializer ---
        log.info("Saving model using SDZSerializer to ZIP archive...");
        long startTime = System.currentTimeMillis();
        SDZSerializer.save(sd, targetZipFile, true, Collections.singletonMap("TestType", "LargeModelZip")); // Use SDZSerializer
        long endTime = System.currentTimeMillis();
        log.info("SDZSerializer save completed in {} ms.", (endTime - startTime));

        // --- Verification ---
        assertTrue(targetZipFile.exists(), "Output ZIP file should exist");
        assertTrue(isZipFile(targetZipFile), "Output file should be a valid ZIP");
        long zipSizeBytes = targetZipFile.length();
        log.info("Actual ZIP file size: {:.2f} GB ({} bytes)", zipSizeBytes / (1024.0 * 1024.0 * 1024.0), zipSizeBytes);
        assertTrue(zipSizeBytes > 1024 * 1024, "ZIP file size should be substantial"); // Basic sanity check

        // Verify internal structure (optional but good)
        try(ZipFile zf = new ZipFile(targetZipFile)) {
            assertTrue(zf.size() > 0, "ZIP file should not be empty");
            boolean hasModelEntry = zf.stream().anyMatch(entry -> entry.getName().startsWith("model."));
            assertTrue(hasModelEntry, "ZIP file must contain at least one entry starting with 'model.'");
            // Optionally check for shard pattern if expected
            boolean looksSharded = zf.stream().anyMatch(entry -> entry.getName().matches("model\\.shard\\d+-of-\\d+\\.sdnb"));
            log.info("ZIP archive appears to contain {} internal shards.", looksSharded ? "multiple" : "a single");
        }

        // --- Load the model back using the correct SDZSerializer ---
        log.info("Attempting to load model using SDZSerializer from ZIP archive...");
        startTime = System.currentTimeMillis();
        SameDiff loadedModel = SDZSerializer.load(targetZipFile, true); // Use SDZSerializer
        endTime = System.currentTimeMillis();
        log.info("SDZSerializer load completed in {} ms.", (endTime - startTime));


        // --- Post-load Verification ---
        assertNotNull(loadedModel, "Loaded model should not be null");
        assertEquals(originalVariableNames.size(), loadedModel.variables().size(), "Variable count mismatch");
        assertEquals(originalOpNames.size(), loadedModel.getOps().size(), "Op count mismatch");
        assertEquals(originalVariableNames, loadedModel.variableNames().stream().collect(Collectors.toSet()), "Variable names mismatch");

        // Verify shapes and data types using the collected maps
        for (SDVariable var : loadedModel.variables()) {
            String name = var.name();
            assertNotNull(name, "Loaded variable has null name");
            assertTrue(originalDataTypes.containsKey(name), "Original data type missing for loaded var " + name);
            assertEquals(originalDataTypes.get(name), var.dataType(), "Data type mismatch for variable " + name);
        }
        assertEquals(originalOpNames, loadedModel.getOps().values().stream().map(SameDiffOp::getName).collect(Collectors.toSet()), "Operation names mismatch");


        // Verify array contents for a subset of variables (as before)
        List<String> samplesToCheck = Arrays.asList(
                "layer_0_w", "layer_0_b",
                "layer_" + Math.min(numLayers-1, numLayers/3) + "_w",
                "layer_" + Math.min(numLayers-1, 2*numLayers/3) + "_b",
                "output_w", "output_b"
        );
        double epsilon = 1e-5;
        for (String varName : samplesToCheck) {
            if (!originalArrays.containsKey(varName)) { log.warn("Skipping check for sample '{}', key missing in original map", varName); continue; }
            INDArray originalArray = originalArrays.get(varName);
            assertNotNull(originalArray, "Original array map contains null for " + varName);
            SDVariable loadedVar = loadedModel.getVariable(varName); assertNotNull(loadedVar, "Loaded var is null for " + varName);
            INDArray loadedArray = loadedVar.getArr(); assertNotNull(loadedArray, "Loaded array is null for " + varName);

            assertEquals(originalArray.dataType(), loadedArray.dataType(), "Array data type mismatch for " + varName);
            assertArrayEquals(originalArray.shape(), loadedArray.shape(), "Array shape mismatch for " + varName);
            assertEquals(originalArray.sumNumber().doubleValue(), loadedArray.sumNumber().doubleValue(), epsilon * originalArray.length(), "Array sum mismatch for " + varName);
            assertEquals(originalArray.minNumber().doubleValue(), loadedArray.minNumber().doubleValue(), epsilon, "Array min mismatch for " + varName);
            assertEquals(originalArray.maxNumber().doubleValue(), loadedArray.maxNumber().doubleValue(), epsilon, "Array max mismatch for " + varName);
            assertTrue(originalArray.equalsWithEps(loadedArray, epsilon),
                    () -> "Array contents mismatch for " + varName + ". Max diff: " + originalArray.sub(loadedArray).amaxNumber());
            log.debug("Verified array content for {}", varName);
        }
        log.info("Array content verification passed for samples.");

        log.info("Test testLargeModelSerializationZip completed successfully!");
    }


    @Test
    //@Ignore // Uncomment if test consistently fails due to resource limits
    public void testMultipleDataTypeSerializationZip(@TempDir Path tempDir) throws IOException {
        // --- This test verifies the ZIP format (.sdz) ---
        // --- It should use SDZSerializer ---

        // Parameters for model with multiple data types
        int numLayers = 3;
        int layerSize = 1000;

        // Assume enough memory
        assumeTrue(hasEnoughMemory(500L * 1024 * 1024), "Skipping multi-datatype ZIP test: Insufficient memory estimated"); // Rough estimate

        // Configure memory
        Nd4j.getEnvironment().setCudaDeviceLimit(Environment.CUDA_LIMIT_MALLOC_HEAP_SIZE, 2L * 1024 * 1024 * 1024); // 2GB limit
        System.out.println("Current malloc heap size limit: " + Nd4j.getEnvironment().cudaMallocHeapSize());

        // Set up temp file
        File tempZipFile = tempDir.resolve("multi-datatype-samediff-model.sdz").toFile();
        log.info("Will save multi-datatype model (.sdz format) to: {}", tempZipFile.getAbsolutePath());

        // Create the model (same logic as original test)
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, layerSize);
        SDVariable current = input;
        DataType[] layerTypes = { DataType.FLOAT, DataType.DOUBLE, DataType.HALF };
        char[] layerOrders = {'c', 'f', 'c'};
        Map<String, INDArray> originalArrays = new HashMap<>();
        System.out.println("Creating neural network with multiple data types...");
        for (int i = 0; i < numLayers; i++) {
            String layerName = "layer_" + i;
            DataType dtype = layerTypes[i];
            char order = layerOrders[i];

            System.out.println("Creating layer " + i + " with data type " + dtype + " and order " + order);

            INDArray weightArray; INDArray biasArray;
            if (dtype == DataType.FLOAT || dtype == DataType.DOUBLE) {
                weightArray = Nd4j.rand(dtype, layerSize, layerSize).dup(order).muli(0.1);
                biasArray = Nd4j.rand(dtype, 1, layerSize).dup(order).muli(0.01);
            } else if (dtype == DataType.HALF) {
                weightArray = Nd4j.rand(DataType.FLOAT, layerSize, layerSize).dup(order).muli(0.1).castTo(DataType.HALF);
                biasArray = Nd4j.rand(DataType.FLOAT, 1, layerSize).dup(order).muli(0.01).castTo(DataType.HALF);
            } else {
                throw new UnsupportedOperationException("Integer types not fully implemented in this test setup");
            }

            SDVariable weights = sd.var(layerName + "_w", weightArray);
            SDVariable bias = sd.var(layerName + "_b", biasArray);
            originalArrays.put(layerName + "_w", weightArray.dup());
            originalArrays.put(layerName + "_b", biasArray.dup());

            SDVariable layerInput = current;
            if (i > 0 && layerInput.dataType() != dtype) {
                layerInput = layerInput.castTo(dtype);
            }
            SDVariable z = layerInput.mmul(weights).add(bias);
            current = sd.nn().relu(z, 0.0);
        }
        SDVariable outputWeights = sd.var("output_w", Nd4j.rand(DataType.FLOAT, layerSize, 10).muli(0.1));
        SDVariable outputBias = sd.var("output_b", Nd4j.rand(DataType.FLOAT, 1, 10).muli(0.01));
        originalArrays.put("output_w", outputWeights.getArr().dup());
        originalArrays.put("output_b", outputBias.getArr().dup());
        if (current.dataType() != DataType.FLOAT) current = current.castTo(DataType.FLOAT);
        SDVariable output = sd.nn().softmax(current.mmul(outputWeights).add(outputBias));
        sd.setOutputs(output.name());
        System.out.println("Multi-datatype model creation finished.");


        // *** CORRECTED: Save the model TO ZIP using SDZSerializer ***
        System.out.println("Saving model to ZIP archive using SDZSerializer...");
        long startTime = System.currentTimeMillis();
        SDZSerializer.save(sd, tempZipFile, true, Collections.emptyMap()); // USE SDZSerializer
        long endTime = System.currentTimeMillis();
        System.out.println("Save completed in " + (endTime - startTime) + " ms");

        // Verify ZIP file was created
        assertTrue(tempZipFile.exists(), "Model ZIP file should exist");
        assertTrue(isZipFile(tempZipFile), "File should be a ZIP");

        // *** CORRECTED: Load the model back FROM ZIP using SDZSerializer ***
        System.out.println("Loading model from ZIP archive using SDZSerializer...");
        startTime = System.currentTimeMillis();
        SameDiff loadedModel = SDZSerializer.load(tempZipFile, true); // USE SDZSerializer
        endTime = System.currentTimeMillis();
        System.out.println("Load completed in " + (endTime - startTime) + " ms");

        // Basic validation (same as before)
        assertNotNull(loadedModel, "Loaded model should not be null");
        assertEquals(sd.variables().size(), loadedModel.variables().size(), "Variable count mismatch");
        assertEquals(sd.ops().length, loadedModel.ops().length, "Op count mismatch");
        assertEquals(sd.variableNames().stream().collect(Collectors.toSet()), loadedModel.variableNames().stream().collect(Collectors.toSet()), "Variable names mismatch");


        // Verify all layers (using helper)
        for (int i = 0; i < numLayers; i++) {
            String layerName = "layer_" + i;
            verifyVariable(loadedModel, originalArrays, layerName + "_w", layerTypes[i], layerOrders[i], layerSize, layerSize);
            verifyVariable(loadedModel, originalArrays, layerName + "_b", layerTypes[i], layerOrders[i], 1, layerSize);
        }
        // Verify output layer
        verifyVariable(loadedModel, originalArrays, "output_w", DataType.FLOAT, 'c', layerSize, 10);
        verifyVariable(loadedModel, originalArrays, "output_b", DataType.FLOAT, 'c', 1, 10);

        System.out.println("Multi-datatype ZIP serialization test completed successfully");
    }

    // --- Helpers ---

    /** Checks if a file is likely a ZIP archive based on magic number. */
    private static boolean isZipFile(File file) {
        if (file == null || !file.exists() || !file.isFile() || file.length() < 4) return false;
        byte[] magic = new byte[4];
        try (FileInputStream fis = new FileInputStream(file); DataInputStream dis = new DataInputStream(fis)) { dis.readFully(magic); } catch (IOException e) { return false; }
        // Standard ZIP magic number PK\03\04 (50 4B 03 04)
        return magic[0] == 0x50 && magic[1] == 0x4b && magic[2] == 0x03 && magic[3] == 0x04;
    }

    /** Helper to verify a variable's properties */
    private void verifyVariable(SameDiff loadedModel, Map<String, INDArray> originalArrays,
                                String varName, DataType expectedType, char expectedOrder,
                                long... expectedShape) { // Use varargs for shape consistency
        assertTrue(loadedModel.hasVariable(varName), "Missing variable in loaded model: " + varName);
        SDVariable loadedVar = loadedModel.getVariable(varName);
        assertNotNull(loadedVar, "Loaded SDVariable is null for " + varName);
        INDArray loadedArray = loadedVar.getArr();
        assertNotNull(loadedArray, "Loaded array (getArr) is null for " + varName);

        assertTrue(originalArrays.containsKey(varName), "Missing variable in original map: " + varName);
        INDArray originalArray = originalArrays.get(varName);
        assertNotNull(originalArray, "Original array is null in map for " + varName);

        assertEquals(expectedType, loadedArray.dataType(), "Data type mismatch for " + varName);
        // Note: Verifying 'order' can be tricky due to internal copies/views. Focus on shape and content.
        // assertEquals(expectedOrder, loadedArray.ordering(), "Ordering mismatch for " + varName);
        assertArrayEquals(expectedShape, loadedArray.shape(), "Shape mismatch for " + varName);
        assertTrue(originalArray.equalsWithEps(loadedArray, 1e-5),
                () -> "Content mismatch for " + varName + ". Max diff: " + originalArray.sub(loadedArray).amaxNumber());
    }




    /** Helper to verify a variable's properties (same as original test) */
    private void verifyVariable(SameDiff loadedModel, Map<String, INDArray> originalArrays,
                                String varName, DataType expectedType, char expectedOrder,
                                int expectedRows, int expectedCols) {
        // (Implementation unchanged from original test)
        assertTrue(loadedModel.hasVariable(varName), "Missing variable: " + varName);
        INDArray originalArray = originalArrays.get(varName);
        INDArray loadedArray = loadedModel.getVariable(varName).getArr();
        assertNotNull(originalArray, "Original array should not be null for " + varName);
        assertNotNull(loadedArray, "Loaded array should not be null for " + varName);
        assertEquals(expectedType, loadedArray.dataType(), "Data type mismatch for " + varName);
        // Ordering check might be less reliable if arrays are small/reshaped, focus on data
        // assertEquals(expectedOrder, loadedArray.ordering(), "Ordering mismatch for " + varName);
        assertArrayEquals(new long[]{expectedRows, expectedCols}, loadedArray.shape(), "Shape mismatch for " + varName);
        assertTrue(originalArray.equalsWithEps(loadedArray, 1e-5), "Content mismatch for " + varName);

    }

    private long moveManifestPast2GB(File shardFile) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(shardFile, "rw")) {
            raf.seek(8); // SDNB magic (4 bytes) + version (4 bytes)
            long manifestOffset = raf.readLong();
            long manifestLength = raf.readLong();
            long metadataOffset = raf.readLong();

            assertTrue(manifestLength > 0, "Manifest must be present for this regression");
            assertTrue(manifestLength < Integer.MAX_VALUE, "Manifest must be small enough for the test helper");
            assertTrue(manifestOffset > metadataOffset, "Shard must contain appended data before the manifest");

            byte[] manifestBytes = new byte[(int) manifestLength];
            raf.seek(manifestOffset);
            raf.readFully(manifestBytes);

            long sparseManifestOffset = (long) Integer.MAX_VALUE + metadataOffset + 4096L;
            raf.seek(sparseManifestOffset);
            raf.write(manifestBytes);
            raf.seek(8);
            raf.writeLong(sparseManifestOffset);
            raf.setLength(sparseManifestOffset + manifestLength);
            return sparseManifestOffset - metadataOffset;
        }
    }

    private void assertShardManifestMagic(File shardFile) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(shardFile, "r")) {
            byte[] magic = new byte[4];
            raf.readFully(magic);
            assertArrayEquals(new byte[]{'S', 'D', 'N', 'B'}, magic, "SDNB magic mismatch");
            assertEquals(1, raf.readInt(), "Unexpected SDNB version");
            long manifestOffset = raf.readLong();
            long manifestLength = raf.readLong();
            long metadataOffset = raf.readLong();

            assertEquals(32L, metadataOffset, "Unexpected SDNB metadata offset");
            assertTrue(manifestLength >= 4, "Manifest length must include Java serialization header");
            assertTrue(manifestOffset >= metadataOffset, "Manifest offset must be inside shard");
            assertTrue(manifestOffset + manifestLength <= raf.length(), "Manifest range exceeds file");

            byte[] manifestHeader = new byte[4];
            raf.seek(manifestOffset);
            raf.readFully(manifestHeader);
            assertArrayEquals(new byte[]{(byte) 0xAC, (byte) 0xED, 0x00, 0x05}, manifestHeader,
                    "Manifest must start with Java serialization stream magic");
        }
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
}