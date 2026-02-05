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

package org.eclipse.deeplearning4j.nd4j.linalg;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

/**
 * Reproducer tests for heap corruption ("double free or corruption (!prev)")
 * observed during large ONNX model import.
 *
 * Observations from crash:
 * - Crash at allocation #13875, total_bytes ~5.2GB
 * - Last ops before crash: scatter_nd_update, gather, add
 * - compute-sanitizer: 0 CUDA errors (CPU-side corruption)
 * - GC disabled: still crashes (NOT a deallocation race)
 * - Smaller models work fine
 *
 * Key insight: GC disabled still crashes means corruption is DURING
 * allocation/data-copy, not during deallocation. Something writes
 * past the end of a buffer, corrupting malloc chunk headers.
 *
 * These tests use SMALL tensors to avoid GPU OOM. The focus is on
 * allocation COUNT and interaction patterns, not memory volume.
 */
@Slf4j
@NativeTag
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class HeapCorruptionReproducerTest {

    @BeforeEach
    public void setup() {
        // Clear any pending CUDA errors from previous tests
        Nd4j.getExecutioner().commit();
    }

    /**
     * Test 1: Many constants + op execution in SameDiff.
     *
     * Simulates what happens during large ONNX model import:
     * thousands of constants are created, then ops execute that
     * read from those constants and allocate output buffers.
     * Uses SMALL tensors (64-dim) to avoid GPU OOM.
     */
    @Test
    @Order(1)
    @DisplayName("Many constants with op execution")
    public void testManyConstantsWithOpExecution() {
        log.info("=== Test: Many constants with op execution ===");
        SameDiff sd = SameDiff.create();

        int numConstants = 5000;
        int dim = 64;  // Small to avoid GPU OOM

        log.info("Creating {} constants...", numConstants);
        List<SDVariable> constants = new ArrayList<>();
        for (int i = 0; i < numConstants; i++) {
            INDArray arr;
            if (i % 4 == 0) {
                arr = Nd4j.randn(DataType.FLOAT, dim, dim);
            } else if (i % 4 == 1) {
                arr = Nd4j.randn(DataType.FLOAT, dim);
            } else if (i % 4 == 2) {
                arr = Nd4j.scalar(DataType.INT64, i);
            } else {
                arr = Nd4j.randn(DataType.FLOAT, 4, 16);
            }
            SDVariable c = sd.constant("const_" + i, arr);
            constants.add(c);

            if (i % 1000 == 0 && i > 0) {
                log.info("  Created {} constants", i);
            }
        }
        log.info("Created {} constants", numConstants);

        // Chain of matmul + add ops
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, dim);
        SDVariable current = input;
        int numLayers = 12;
        for (int layer = 0; layer < numLayers; layer++) {
            int wIdx = layer * 4;
            int bIdx = layer * 4 + 1;
            if (wIdx >= numConstants || bIdx >= numConstants) break;
            current = sd.mmul("layer_" + layer + "_mm", current, constants.get(wIdx));
            current = current.add("layer_" + layer + "_add", constants.get(bIdx));
        }

        sd.setOutputs(Collections.singletonList(current.name()));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, dim);
        Map<String, INDArray> output = sd.output(Map.of("input", inputArr), sd.outputs().toArray(new String[0]));

        for (var entry : output.entrySet()) {
            log.info("Output '{}': shape={}", entry.getKey(), Arrays.toString(entry.getValue().shape()));
        }
        log.info("=== Test passed ===");
    }

    /**
     * Test 2: High count buffer allocation stress.
     *
     * Creates 15000+ small buffers (past the crash point of 13875)
     * then runs ops. Focus is on allocation COUNT, not size.
     */
    @Test
    @Order(2)
    @DisplayName("High count buffer allocation stress")
    public void testHighCountBufferAllocation() {
        log.info("=== Test: High count buffer allocation stress ===");

        int totalAllocations = 15000;
        List<INDArray> keepAlive = new ArrayList<>();

        for (int i = 0; i < totalAllocations; i++) {
            INDArray arr;
            switch (i % 5) {
                case 0: arr = Nd4j.zeros(DataType.FLOAT, 32, 32); break;
                case 1: arr = Nd4j.zeros(DataType.FLOAT, 64); break;
                case 2: arr = Nd4j.scalar(DataType.INT64, i); break;
                case 3: arr = Nd4j.zeros(DataType.FLOAT, 4, 16); break;
                default: arr = Nd4j.zeros(DataType.FLOAT, 16, 16); break;
            }
            keepAlive.add(arr);

            if (i % 3000 == 0 && i > 0) {
                log.info("  Allocated {} buffers", i);
            }
        }

        log.info("Allocated {} buffers, now running ops...", totalAllocations);

        // Run ops on some of them to trigger additional output buffer allocations
        for (int i = 0; i < 200; i++) {
            int idx = (i * 50) % keepAlive.size();
            INDArray a = keepAlive.get(idx);
            if (a.length() > 1) {
                INDArray result = a.add(1.0);
                result.close();
            }
        }

        log.info("Ops completed");
        for (INDArray arr : keepAlive) {
            arr.close();
        }
        log.info("=== Test passed ===");
    }

    /**
     * Test 3: Gather + Add with many constants.
     *
     * Directly mimics the op sequence seen before the crash:
     * gather, add with many constants in the graph.
     */
    @Test
    @Order(3)
    @DisplayName("Gather/Add with many constants")
    public void testGatherAddWithManyConstants() {
        log.info("=== Test: Gather/Add with many constants ===");
        SameDiff sd = SameDiff.create();

        int numConstants = 3000;
        log.info("Creating {} constants...", numConstants);
        for (int i = 0; i < numConstants; i++) {
            if (i % 3 == 0) {
                sd.constant("w_" + i, Nd4j.randn(DataType.FLOAT, 32, 32));
            } else if (i % 3 == 1) {
                sd.constant("b_" + i, Nd4j.randn(DataType.FLOAT, 32));
            } else {
                sd.constant("s_" + i, Nd4j.scalar(DataType.INT64, i));
            }
        }

        int vocabSize = 256;
        int hiddenSize = 32;
        int seqLen = 16;

        SDVariable embeddings = sd.constant("embeddings", Nd4j.randn(DataType.FLOAT, vocabSize, hiddenSize));
        SDVariable inputIds = sd.placeHolder("input_ids", DataType.INT64, -1, seqLen);

        SDVariable gathered = sd.gather("gather_embed", embeddings, inputIds, 0);
        SDVariable bias = sd.constant("residual_bias", Nd4j.randn(DataType.FLOAT, hiddenSize));
        SDVariable added = gathered.add("add_residual", bias);

        sd.setOutputs(Collections.singletonList(added.name()));

        INDArray ids = Nd4j.zeros(DataType.INT64, 1, seqLen);
        for (int i = 0; i < seqLen; i++) {
            ids.putScalar(new long[]{0, i}, i % vocabSize);
        }

        Map<String, INDArray> output = sd.output(Map.of("input_ids", ids), sd.outputs().toArray(new String[0]));
        for (var entry : output.entrySet()) {
            log.info("Output '{}': shape={}", entry.getKey(), Arrays.toString(entry.getValue().shape()));
        }
        log.info("=== Test passed ===");
    }

    /**
     * Test 4: Repeated forward passes with constant-heavy graph.
     *
     * The VLM test runs the vision encoder multiple times (once per tile).
     * Each pass uses the same constants but creates new intermediate buffers.
     */
    @Test
    @Order(4)
    @DisplayName("Repeated forward passes with many constants")
    public void testRepeatedForwardPassesWithConstants() {
        log.info("=== Test: Repeated forward passes with many constants ===");
        SameDiff sd = SameDiff.create();

        int dim = 64;
        int numLayers = 6;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, dim);
        SDVariable current = input;

        for (int layer = 0; layer < numLayers; layer++) {
            SDVariable w = sd.constant("layer_" + layer + "_w", Nd4j.randn(DataType.FLOAT, dim, dim));
            SDVariable b = sd.constant("layer_" + layer + "_b", Nd4j.randn(DataType.FLOAT, dim));
            SDVariable gamma = sd.constant("layer_" + layer + "_gamma", Nd4j.ones(DataType.FLOAT, dim));
            SDVariable beta = sd.constant("layer_" + layer + "_beta", Nd4j.zeros(DataType.FLOAT, dim));

            current = sd.mmul("layer_" + layer + "_mm", current, w);
            current = current.add("layer_" + layer + "_add", b);
        }

        sd.setOutputs(Collections.singletonList(current.name()));

        int numPasses = 20;
        for (int pass = 0; pass < numPasses; pass++) {
            INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, dim);
            Map<String, INDArray> output = sd.output(Map.of("input", inputArr), sd.outputs().toArray(new String[0]));

            if (pass == 0) {
                for (var entry : output.entrySet()) {
                    log.info("Pass {} output: shape={}", pass, Arrays.toString(entry.getValue().shape()));
                }
            }

            inputArr.close();
            sd.clearPlaceholders(false);
            sd.clearOpInputs();
        }

        log.info("Completed {} forward passes", numPasses);
        log.info("=== Test passed ===");
    }

    /**
     * Test 5: Rapid small-op execution with many live buffers.
     *
     * Keep many buffers alive while executing rapid add operations.
     * Exercises the new InteropDataBuffer acquireAccess/releaseAccess
     * paths under contention.
     */
    @Test
    @Order(5)
    @DisplayName("Rapid ops with many live buffers")
    public void testRapidOpsWithManyLiveBuffers() {
        log.info("=== Test: Rapid ops with many live buffers ===");

        int numArrays = 5000;
        INDArray[] arrays = new INDArray[numArrays];
        for (int i = 0; i < numArrays; i++) {
            arrays[i] = Nd4j.randn(DataType.FLOAT, 16, 16);
        }
        log.info("Created {} arrays", numArrays);

        for (int iter = 0; iter < 3; iter++) {
            for (int i = 0; i < numArrays - 1; i++) {
                INDArray result = arrays[i].add(arrays[i + 1]);
                result.close();
            }
            log.info("  Iteration {} complete", iter);
        }

        for (INDArray arr : arrays) {
            arr.close();
        }
        log.info("=== Test passed ===");
    }

    /**
     * Test 6: SameDiff constant flag stress.
     *
     * Multiple SameDiff instances with many constants each,
     * created and executed in sequence. Tests that the new atomic
     * isConstant flag works correctly under load.
     */
    @Test
    @Order(6)
    @DisplayName("SameDiff constant flag stress test")
    public void testSameDiffConstantFlagStress() {
        log.info("=== Test: SameDiff constant flag stress ===");

        for (int trial = 0; trial < 5; trial++) {
            SameDiff sd = SameDiff.create();

            int numConst = 2000;
            for (int i = 0; i < numConst; i++) {
                if (i % 3 == 0) {
                    sd.constant("c_" + i, Nd4j.randn(DataType.FLOAT, 16, 16));
                } else if (i % 3 == 1) {
                    sd.constant("c_" + i, Nd4j.ones(DataType.FLOAT, 16));
                } else {
                    sd.constant("c_" + i, Nd4j.scalar(DataType.INT64, i));
                }
            }

            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
            SDVariable w = sd.getVariable("c_0");
            SDVariable b = sd.getVariable("c_1");
            SDVariable result = sd.mmul("mm", x, w).add("out", b);
            sd.setOutputs(Collections.singletonList("out"));

            Map<String, INDArray> out = sd.output(
                    Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 16)),
                    "out"
            );

            log.info("Trial {}: output shape={}", trial, Arrays.toString(out.get("out").shape()));
        }

        log.info("=== Test passed ===");
    }

    /**
     * Test 7: Mixed dtype constants with ops.
     *
     * Real ONNX models have constants of different dtypes (FLOAT, INT64, BOOL).
     * The allocation paths differ by dtype. Test that mixing dtypes in high
     * volume doesn't corrupt anything.
     */
    @Test
    @Order(7)
    @DisplayName("Mixed dtype constants stress")
    public void testMixedDtypeConstantsStress() {
        log.info("=== Test: Mixed dtype constants stress ===");
        SameDiff sd = SameDiff.create();

        int numConst = 4000;
        DataType[] dtypes = {DataType.FLOAT, DataType.INT64, DataType.BOOL, DataType.DOUBLE, DataType.INT32};

        for (int i = 0; i < numConst; i++) {
            DataType dt = dtypes[i % dtypes.length];
            INDArray arr;
            if (dt == DataType.FLOAT) {
                arr = Nd4j.zeros(DataType.FLOAT, 16, 16);
            } else if (dt == DataType.INT64) {
                arr = Nd4j.zeros(DataType.INT64, 8);
            } else if (dt == DataType.BOOL) {
                arr = Nd4j.zeros(DataType.BOOL, 4);
            } else if (dt == DataType.DOUBLE) {
                arr = Nd4j.zeros(DataType.DOUBLE, 16);
            } else if (dt == DataType.INT32) {
                arr = Nd4j.zeros(DataType.INT32, 4, 4);
            } else {
                arr = Nd4j.scalar(0.0f);
            }
            sd.constant("mc_" + i, arr);
        }

        log.info("Created {} mixed-dtype constants", numConst);

        // Simple graph using float constants
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w = sd.constant("final_w", Nd4j.randn(DataType.FLOAT, 16, 16));
        SDVariable out = sd.mmul("mm_out", x, w);
        sd.setOutputs(Collections.singletonList("mm_out"));

        Map<String, INDArray> result = sd.output(
                Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 16)),
                "mm_out"
        );

        log.info("Output shape={}", Arrays.toString(result.get("mm_out").shape()));
        log.info("=== Test passed ===");
    }

    /**
     * Test 8: Large buffer allocations with many small constants.
     *
     * The original crash was at ~5.2GB total allocated with ~13875 buffers.
     * This test creates a mix of large and small buffers to hit the memory
     * volume threshold. Large buffers are more likely to cause corruption
     * if an op writes past the end.
     */
    @Test
    @Order(8)
    @DisplayName("Large buffers mixed with many small constants")
    public void testLargeBuffersWithManySmallConstants() {
        log.info("=== Test: Large buffers mixed with many small constants ===");
        SameDiff sd = SameDiff.create();

        // Create many small constants first (simulates ONNX model metadata)
        int numSmallConst = 2000;
        for (int i = 0; i < numSmallConst; i++) {
            if (i % 3 == 0) {
                sd.constant("s_" + i, Nd4j.zeros(DataType.FLOAT, 8));
            } else if (i % 3 == 1) {
                sd.constant("s_" + i, Nd4j.scalar(DataType.INT64, i));
            } else {
                sd.constant("s_" + i, Nd4j.zeros(DataType.BOOL, 4));
            }
        }
        log.info("Created {} small constants", numSmallConst);

        // Now create larger buffers (simulates embedding tables, weight matrices)
        int hiddenSize = 576;  // Typical VLM hidden size
        int numLargeConst = 50;
        for (int i = 0; i < numLargeConst; i++) {
            sd.constant("large_" + i, Nd4j.randn(DataType.FLOAT, hiddenSize, hiddenSize));
        }
        log.info("Created {} large constants ({}x{})", numLargeConst, hiddenSize, hiddenSize);

        // Build graph with matmul chain
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, hiddenSize);
        SDVariable current = x;
        for (int i = 0; i < 6; i++) {
            SDVariable w = sd.getVariable("large_" + i);
            current = sd.mmul("mm_" + i, current, w);
        }
        sd.setOutputs(Collections.singletonList(current.name()));

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, hiddenSize);
        Map<String, INDArray> out = sd.output(Map.of("x", input), sd.outputs().toArray(new String[0]));
        log.info("Output shape={}", Arrays.toString(out.values().iterator().next().shape()));

        input.close();
        log.info("=== Test passed ===");
    }

    /**
     * Test 9: Scatter/Gather pattern with large embedding tables.
     *
     * The crash showed scatter_nd_update and gather as last ops.
     * These ops index into large tables which is a common source of
     * out-of-bounds writes if indices are wrong.
     */
    @Test
    @Order(9)
    @DisplayName("Scatter/Gather with large embeddings")
    public void testScatterGatherWithLargeEmbeddings() {
        log.info("=== Test: Scatter/Gather with large embeddings ===");
        SameDiff sd = SameDiff.create();

        int vocabSize = 4096;
        int hiddenSize = 256;
        int seqLen = 128;

        // Large embedding table
        SDVariable embeddings = sd.constant("embed_table",
                Nd4j.randn(DataType.FLOAT, vocabSize, hiddenSize));
        SDVariable posEmbed = sd.constant("pos_embed",
                Nd4j.randn(DataType.FLOAT, seqLen, hiddenSize));

        // Add many small constants (like a real model)
        for (int i = 0; i < 500; i++) {
            sd.constant("meta_" + i, Nd4j.scalar(DataType.INT64, i));
        }

        SDVariable inputIds = sd.placeHolder("input_ids", DataType.INT64, -1, seqLen);
        SDVariable gathered = sd.gather("gather_embed", embeddings, inputIds, 0);
        SDVariable added = gathered.add("add_pos", posEmbed);

        // Add weight matrices after gather
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, hiddenSize, hiddenSize));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, hiddenSize, hiddenSize));
        SDVariable h = sd.mmul("mm1", added, w1);
        h = sd.mmul("mm2", h, w2);

        sd.setOutputs(Collections.singletonList(h.name()));

        INDArray ids = Nd4j.zeros(DataType.INT64, 1, seqLen);
        Random rng = new Random(42);
        for (int i = 0; i < seqLen; i++) {
            ids.putScalar(new long[]{0, i}, rng.nextInt(vocabSize));
        }

        Map<String, INDArray> out = sd.output(Map.of("input_ids", ids), sd.outputs().toArray(new String[0]));
        log.info("Output shape={}", Arrays.toString(out.values().iterator().next().shape()));
        ids.close();
        log.info("=== Test passed ===");
    }

    /**
     * Test 10: Cumulative memory pressure - allocate/free/reallocate cycle.
     *
     * The real crash happens at high cumulative memory (~5.2GB).
     * This test allocates, frees, and reallocates to accumulate
     * total bytes allocated while keeping peak memory manageable.
     * Tests that freed memory doesn't corrupt heap metadata.
     */
    @Test
    @Order(10)
    @DisplayName("Cumulative memory pressure with alloc/free cycles")
    public void testCumulativeMemoryPressure() {
        log.info("=== Test: Cumulative memory pressure ===");

        long totalBytesAllocated = 0;
        long targetBytes = 4L * 1024 * 1024 * 1024; // 4GB cumulative
        int batchSize = 200;
        int iteration = 0;

        while (totalBytesAllocated < targetBytes) {
            List<INDArray> batch = new ArrayList<>();
            for (int i = 0; i < batchSize; i++) {
                int size = 256 + (i % 10) * 64;
                INDArray arr = Nd4j.randn(DataType.FLOAT, size, size);
                batch.add(arr);
                totalBytesAllocated += (long) size * size * 4;
            }

            // Run some ops on them
            for (int i = 0; i < batch.size() - 1; i++) {
                if (batch.get(i).shape()[0] == batch.get(i + 1).shape()[0]) {
                    INDArray result = batch.get(i).add(batch.get(i + 1));
                    result.close();
                }
            }

            // Free all
            for (INDArray arr : batch) {
                arr.close();
            }

            iteration++;
            if (iteration % 5 == 0) {
                log.info("  Iteration {}: cumulative {}MB allocated",
                        iteration, totalBytesAllocated / (1024 * 1024));
            }
        }

        log.info("Completed {} iterations, {}GB cumulative allocated",
                iteration, totalBytesAllocated / (1024 * 1024 * 1024));
        log.info("=== Test passed ===");
    }

    /**
     * Test 11: Direct ByteBuffer → createBufferCpuOnly path.
     *
     * This is the EXACT code path used by OnnxIRTensor.loadOnnxRawDataDirect().
     * Creates direct ByteBuffers, fills with data, then creates CPU-only buffers
     * and reshapes. This exercises:
     * - ByteBuffer.allocateDirect
     * - Nd4j.createBufferCpuOnly(byteBuffer, dtype, totalLen)
     * - Nd4j.create(buffer).reshape('c', shape)
     */
    @Test
    @Order(11)
    @DisplayName("ByteBuffer createBufferCpuOnly import path")
    public void testByteBufferCpuOnlyImportPath() {
        log.info("=== Test: ByteBuffer createBufferCpuOnly import path ===");

        int numConstants = 1500;
        List<INDArray> constants = new ArrayList<>();

        for (int i = 0; i < numConstants; i++) {
            DataType dtype;
            long[] shape;
            if (i % 5 == 0) {
                dtype = DataType.FLOAT;
                shape = new long[]{256, 256};
            } else if (i % 5 == 1) {
                dtype = DataType.FLOAT;
                shape = new long[]{256};
            } else if (i % 5 == 2) {
                dtype = DataType.INT64;
                shape = new long[]{1};
            } else if (i % 5 == 3) {
                dtype = DataType.FLOAT;
                shape = new long[]{4, 256};
            } else {
                dtype = DataType.FLOAT;
                shape = new long[]{16, 16};
            }

            long totalLen = 1;
            for (long dim : shape) totalLen *= dim;

            // Mimic loadOnnxRawDataDirect: allocate direct buffer, fill, create cpu-only
            int byteSize = (int) (totalLen * dtype.width());
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(byteSize);
            byteBuffer.order(ByteOrder.LITTLE_ENDIAN);

            // Fill with pseudo-data (like protobuf raw_data)
            Random rng = new Random(i);
            for (int b = 0; b < byteSize; b++) {
                byteBuffer.put((byte) rng.nextInt(256));
            }
            ((Buffer) byteBuffer).rewind();

            // This is the exact call from OnnxIRTensor.loadOnnxRawDataDirect
            org.nd4j.linalg.api.buffer.DataBuffer rawDataBuffer =
                    Nd4j.createBufferCpuOnly(byteBuffer, dtype, totalLen);

            INDArray arr;
            if (shape.length > 0 && rawDataBuffer.length() > 0) {
                arr = Nd4j.create(rawDataBuffer).reshape('c', shape);
            } else {
                arr = Nd4j.create(rawDataBuffer);
            }
            constants.add(arr);

            if (i % 500 == 0 && i > 0) {
                log.info("  Created {} constants via ByteBuffer path", i);
            }
        }

        log.info("Created {} constants via ByteBuffer path", numConstants);

        // Now use them in a SameDiff graph (like import does)
        SameDiff sd = SameDiff.create();
        for (int i = 0; i < constants.size(); i++) {
            sd.constant("import_const_" + i, constants.get(i));
        }

        // Build simple graph
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 256);
        SDVariable w = sd.getVariable("import_const_0"); // 256x256
        SDVariable result = sd.mmul("mm", x, w);
        sd.setOutputs(Collections.singletonList("mm"));

        Map<String, INDArray> out = sd.output(
                Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 256)),
                "mm"
        );
        log.info("Output shape={}", Arrays.toString(out.get("mm").shape()));
        log.info("=== Test passed ===");
    }

    /**
     * Test 12: createBufferCpuOnly + reshape + op execution at high volume.
     *
     * Creates buffers the way the ONNX importer does (CPU-only),
     * then triggers GPU migration by running ops. This tests whether
     * lazy GPU migration corrupts anything.
     */
    @Test
    @Order(12)
    @DisplayName("CPU-only buffer lazy GPU migration stress")
    public void testCpuOnlyBufferLazyGpuMigration() {
        log.info("=== Test: CPU-only buffer lazy GPU migration stress ===");

        int numBuffers = 1000;
        List<INDArray> cpuOnlyArrays = new ArrayList<>();

        // Create buffers via CPU-only path
        for (int i = 0; i < numBuffers; i++) {
            int size = 64 + (i % 8) * 32;
            long totalLen = (long) size * size;
            int byteSize = (int) (totalLen * DataType.FLOAT.width());

            ByteBuffer bb = ByteBuffer.allocateDirect(byteSize);
            bb.order(ByteOrder.LITTLE_ENDIAN);
            Random rng = new Random(i);
            for (int b = 0; b < Math.min(byteSize, 4096); b++) {
                bb.put((byte) rng.nextInt(256));
            }
            // Fill rest with zeros if larger
            for (int b = 4096; b < byteSize; b++) {
                bb.put((byte) 0);
            }
            ((Buffer) bb).rewind();

            org.nd4j.linalg.api.buffer.DataBuffer buf =
                    Nd4j.createBufferCpuOnly(bb, DataType.FLOAT, totalLen);
            INDArray arr = Nd4j.create(buf).reshape('c', new long[]{size, size});
            cpuOnlyArrays.add(arr);
        }
        log.info("Created {} CPU-only buffers", numBuffers);

        // Now force GPU migration by running ops
        log.info("Forcing GPU migration via ops...");
        for (int i = 0; i < cpuOnlyArrays.size() - 1; i++) {
            INDArray a = cpuOnlyArrays.get(i);
            INDArray b = cpuOnlyArrays.get(i + 1);
            if (a.shape()[0] == b.shape()[0] && a.shape()[1] == b.shape()[1]) {
                INDArray result = a.add(b);
                result.close();
            }
        }
        log.info("GPU migration ops completed");

        // Clean up
        for (INDArray arr : cpuOnlyArrays) {
            arr.close();
        }
        log.info("=== Test passed ===");
    }

    /**
     * Test 13: Verify createBufferCpuOnly produces correct shapes.
     *
     * The ByteBuffer → createBufferCpuOnly → Nd4j.create → reshape path
     * is the exact path used by OnnxIRTensor.loadOnnxRawDataDirect().
     * This test verifies buffer lengths and shapes are correct.
     */
    @Test
    @Order(13)
    @DisplayName("Verify createBufferCpuOnly shape correctness")
    public void testCreateBufferCpuOnlyShapeCorrectness() {
        log.info("=== Test: Verify createBufferCpuOnly shape correctness ===");

        // Test 1: 1D float buffer via CPU-only path
        // NOTE: reshape('c', 256) in Java resolves to reshape(long...) with args {99, 256}
        // because 'c' gets widened to long. Must use reshape(new long[]{256}) or
        // reshape('c', new long[]{256}) to get correct behavior.
        {
            int len = 256;
            ByteBuffer bb = ByteBuffer.allocateDirect(len * 4);
            bb.order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < len; i++) bb.putFloat(i * 0.1f);
            ((Buffer) bb).rewind();
            org.nd4j.linalg.api.buffer.DataBuffer buf =
                    Nd4j.createBufferCpuOnly(bb, DataType.FLOAT, len);
            log.info("CPU-only 1D buffer: length={}", buf.length());
            Assertions.assertEquals(len, buf.length(), "Buffer length mismatch for 1D float");
            INDArray arr = Nd4j.create(buf);
            log.info("CPU-only 1D array: shape={}, length={}", Arrays.toString(arr.shape()), arr.length());

            // Use long[] to avoid char-widening ambiguity
            INDArray reshaped = arr.reshape('c', new long[]{len});
            log.info("CPU-only 1D reshaped: shape={}", Arrays.toString(reshaped.shape()));
            Assertions.assertArrayEquals(new long[]{len}, reshaped.shape(), "Reshape to [256] failed");
            arr.close();
        }

        // Test 2: 2D float buffer
        {
            int rows = 256, cols = 256;
            long totalLen = (long) rows * cols;
            ByteBuffer bb = ByteBuffer.allocateDirect((int)(totalLen * 4));
            bb.order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < totalLen; i++) bb.putFloat(i * 0.01f);
            ((Buffer) bb).rewind();
            org.nd4j.linalg.api.buffer.DataBuffer buf =
                    Nd4j.createBufferCpuOnly(bb, DataType.FLOAT, totalLen);
            log.info("2D buffer: length={}", buf.length());
            Assertions.assertEquals(totalLen, buf.length(), "Buffer length mismatch for 2D float");
            INDArray arr = Nd4j.create(buf).reshape('c', new long[]{rows, cols});
            log.info("2D array: shape={}", Arrays.toString(arr.shape()));
            Assertions.assertArrayEquals(new long[]{rows, cols}, arr.shape(), "Reshape to [256,256] failed");
            arr.close();
        }

        // Test 3: INT64 scalar
        {
            ByteBuffer bb = ByteBuffer.allocateDirect(8);
            bb.order(ByteOrder.LITTLE_ENDIAN);
            bb.putLong(42L);
            ((Buffer) bb).rewind();
            org.nd4j.linalg.api.buffer.DataBuffer buf =
                    Nd4j.createBufferCpuOnly(bb, DataType.INT64, 1);
            log.info("INT64 scalar buffer: length={}", buf.length());
            Assertions.assertEquals(1, buf.length(), "Buffer length mismatch for INT64 scalar");
            INDArray arr = Nd4j.create(buf);
            log.info("INT64 scalar array: shape={}, length={}", Arrays.toString(arr.shape()), arr.length());
            arr.close();
        }

        // Test 4: Now build a SameDiff graph using CPU-only buffers
        {
            SameDiff sd = SameDiff.create();
            int dim = 64;

            // Weight via CPU-only path
            long wLen = (long) dim * dim;
            ByteBuffer wBuf = ByteBuffer.allocateDirect((int)(wLen * 4));
            wBuf.order(ByteOrder.LITTLE_ENDIAN);
            Random rng = new Random(0);
            for (int i = 0; i < wLen; i++) wBuf.putFloat((float)(rng.nextGaussian() * 0.01));
            ((Buffer) wBuf).rewind();
            INDArray wArr = Nd4j.create(Nd4j.createBufferCpuOnly(wBuf, DataType.FLOAT, wLen))
                    .reshape('c', new long[]{dim, dim});
            log.info("Weight shape: {}", Arrays.toString(wArr.shape()));
            Assertions.assertArrayEquals(new long[]{dim, dim}, wArr.shape());
            sd.constant("w", wArr);

            // Bias via CPU-only path
            ByteBuffer bBuf = ByteBuffer.allocateDirect(dim * 4);
            bBuf.order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < dim; i++) bBuf.putFloat(0.0f);
            ((Buffer) bBuf).rewind();
            INDArray bArr = Nd4j.create(Nd4j.createBufferCpuOnly(bBuf, DataType.FLOAT, dim))
                    .reshape('c', new long[]{dim});
            log.info("Bias shape: {}", Arrays.toString(bArr.shape()));
            Assertions.assertArrayEquals(new long[]{dim}, bArr.shape());
            sd.constant("b", bArr);

            // Build graph
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, dim);
            SDVariable w = sd.getVariable("w");
            SDVariable b = sd.getVariable("b");
            SDVariable result = sd.mmul("mm", x, w).add("out", b);
            sd.setOutputs(Collections.singletonList("out"));

            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
            Map<String, INDArray> out = sd.output(Map.of("x", input), "out");
            log.info("Output shape: {}", Arrays.toString(out.get("out").shape()));
            Assertions.assertArrayEquals(new long[]{1, dim}, out.get("out").shape());
            input.close();
        }

        log.info("=== Test passed ===");
    }

    /**
     * Test 14: Full ONNX import simulation — cpu-only constants loaded,
     * SameDiff graph built and executed (triggering GPU migration).
     */
    @Test
    @Order(14)
    @DisplayName("Full ONNX import simulation")
    public void testFullOnnxImportSimulation() {
        log.info("=== Test: Full ONNX import simulation ===");

        SameDiff sd = SameDiff.create();
        int hiddenSize = 128;
        int numLayers = 6;

        // Load weight/bias constants via CPU-only ByteBuffer path
        for (int layer = 0; layer < numLayers; layer++) {
            // Weight matrix
            long wLen = (long) hiddenSize * hiddenSize;
            ByteBuffer wBuf = ByteBuffer.allocateDirect((int)(wLen * 4));
            wBuf.order(ByteOrder.LITTLE_ENDIAN);
            Random rng = new Random(layer);
            for (int i = 0; i < wLen; i++) wBuf.putFloat((float)(rng.nextGaussian() * 0.01));
            ((Buffer) wBuf).rewind();
            INDArray wArr = Nd4j.create(Nd4j.createBufferCpuOnly(wBuf, DataType.FLOAT, wLen))
                    .reshape('c', new long[]{hiddenSize, hiddenSize});
            sd.constant("layer" + layer + "_w", wArr);

            // Bias
            ByteBuffer bBuf = ByteBuffer.allocateDirect(hiddenSize * 4);
            bBuf.order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < hiddenSize; i++) bBuf.putFloat(0.0f);
            ((Buffer) bBuf).rewind();
            INDArray bArr = Nd4j.create(Nd4j.createBufferCpuOnly(bBuf, DataType.FLOAT, hiddenSize))
                    .reshape('c', new long[]{hiddenSize});
            sd.constant("layer" + layer + "_b", bArr);
        }

        // Add many scalar constants (axis values, indices, etc.)
        for (int i = 0; i < 500; i++) {
            ByteBuffer sBuf = ByteBuffer.allocateDirect(8);
            sBuf.order(ByteOrder.LITTLE_ENDIAN);
            sBuf.putLong(i);
            ((Buffer) sBuf).rewind();
            sd.constant("scalar_" + i,
                    Nd4j.create(Nd4j.createBufferCpuOnly(sBuf, DataType.INT64, 1)));
        }

        log.info("Created {} constants via CPU-only path", numLayers * 2 + 500);

        // Build matmul + add chain
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, hiddenSize);
        SDVariable current = x;
        for (int layer = 0; layer < numLayers; layer++) {
            SDVariable w = sd.getVariable("layer" + layer + "_w");
            SDVariable b = sd.getVariable("layer" + layer + "_b");
            current = sd.mmul("layer" + layer + "_mm", current, w);
            current = current.add("layer" + layer + "_add", b);
        }
        sd.setOutputs(Collections.singletonList(current.name()));

        // Execute (triggers GPU migration)
        log.info("Executing graph (triggers GPU migration)...");
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, hiddenSize);
        Map<String, INDArray> out = sd.output(Map.of("x", input), sd.outputs().toArray(new String[0]));
        log.info("Output shape={}", Arrays.toString(out.values().iterator().next().shape()));

        // Multiple forward passes
        for (int pass = 0; pass < 10; pass++) {
            input = Nd4j.randn(DataType.FLOAT, 1, hiddenSize);
            out = sd.output(Map.of("x", input), sd.outputs().toArray(new String[0]));
            input.close();
            sd.clearPlaceholders(false);
            sd.clearOpInputs();
        }
        log.info("Completed 10 additional forward passes");
        log.info("=== Test passed ===");
    }

    /**
     * Test 15: Vision encoder pipeline reproducer.
     *
     * Mimics the SmolDocling VLM vision encoder pipeline that crashes
     * during reduce_mean on [1, 1024, 768] at layer_norm1.
     * Steps: patch_embed conv → reshape → transpose → gather(pos_embed) → add → reduce_mean
     *
     * This isolates whether the crash is in reduce_mean itself or
     * heap corruption from an earlier op in the pipeline.
     */
    @Test
    @Order(15)
    @DisplayName("Vision encoder pipeline: embed → transpose → gather → add → reduce_mean")
    public void testVisionEncoderReduceMeanPipeline() {
        log.info("=== Test: Vision encoder pipeline reduce_mean crash reproducer ===");
        SameDiff sd = SameDiff.create();

        int seqLen = 1024;
        int hiddenSize = 768;

        // Create many small constants first (simulates large model with many parameters)
        log.info("Creating small constants to simulate model parameter count...");
        for (int i = 0; i < 3000; i++) {
            if (i % 3 == 0) {
                sd.constant("param_" + i, Nd4j.randn(DataType.FLOAT, 8));
            } else if (i % 3 == 1) {
                sd.constant("param_" + i, Nd4j.scalar(DataType.INT64, i));
            } else {
                sd.constant("param_" + i, Nd4j.zeros(DataType.FLOAT, 4, 4));
            }
        }
        log.info("Created 3000 small constants");

        // Create large embedding-sized constants (like position embedding, weight matrices)
        SDVariable posEmbedWeight = sd.constant("pos_embed_weight",
                Nd4j.randn(DataType.FLOAT, seqLen, hiddenSize));
        SDVariable layerNormWeight = sd.constant("ln_weight",
                Nd4j.ones(DataType.FLOAT, hiddenSize));
        SDVariable layerNormBias = sd.constant("ln_bias",
                Nd4j.zeros(DataType.FLOAT, hiddenSize));

        // Input: simulates patch embeddings output [1, 1024, 768]
        SDVariable patchEmbed = sd.placeHolder("patch_embed", DataType.FLOAT, -1, seqLen, hiddenSize);

        // Position embedding indices [1, 1024]
        SDVariable posIndices = sd.placeHolder("pos_indices", DataType.INT64, -1, seqLen);

        // Gather position embeddings: [1024, 768] gathered by [1, 1024] → [1, 1024, 768]
        SDVariable posEmbed = sd.gather("pos_gather", posEmbedWeight, posIndices, 0);

        // Add position embeddings to patch embeddings
        SDVariable embeddings = patchEmbed.add("embed_add", posEmbed);

        // Layer norm: reduce_mean along last axis [-1]
        // This is where the crash happens in the VLM model
        SDVariable mean = sd.mean("layer_norm_mean", embeddings, false, -1);

        // Subtract mean and multiply by weight + add bias (simplified layer norm)
        SDVariable meanReshaped = sd.reshape("mean_reshaped", mean, 1, seqLen, 1);
        SDVariable centered = embeddings.sub("centered", meanReshaped);
        SDVariable normed = centered.mul("normed", layerNormWeight);
        SDVariable output = normed.add("output", layerNormBias);

        sd.setOutputs(Collections.singletonList(output.name()));

        // Create input data
        INDArray patchEmbedArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenSize);
        INDArray posIndicesArr = Nd4j.zeros(DataType.INT64, 1, seqLen);
        for (int i = 0; i < seqLen; i++) {
            posIndicesArr.putScalar(new long[]{0, i}, i);
        }

        log.info("Running forward pass with patch_embed=[1,{},{}], pos_indices=[1,{}]...",
                seqLen, hiddenSize, seqLen);

        Map<String, INDArray> result = sd.output(
                Map.of("patch_embed", patchEmbedArr, "pos_indices", posIndicesArr),
                sd.outputs().toArray(new String[0])
        );

        log.info("Output shape={}", Arrays.toString(result.get("output").shape()));

        // Run multiple forward passes to stress test allocation/deallocation
        for (int pass = 0; pass < 5; pass++) {
            patchEmbedArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenSize);
            result = sd.output(
                    Map.of("patch_embed", patchEmbedArr, "pos_indices", posIndicesArr),
                    sd.outputs().toArray(new String[0])
            );
            log.info("Pass {} output shape={}", pass, Arrays.toString(result.get("output").shape()));
            patchEmbedArr.close();
            sd.clearPlaceholders(false);
            sd.clearOpInputs();
        }

        posIndicesArr.close();
        log.info("=== Test passed ===");
    }

    /**
     * Test 16: Isolated reduce_mean on [1, 1024, 768].
     *
     * Tests reduce_mean in isolation with the exact same shape
     * that crashes in the VLM model. If this passes but test 15 fails,
     * the corruption comes from an earlier op, not reduce_mean itself.
     */
    @Test
    @Order(16)
    @DisplayName("Isolated reduce_mean on [1, 1024, 768]")
    public void testIsolatedReduceMean() {
        log.info("=== Test: Isolated reduce_mean on [1, 1024, 768] ===");

        // Direct Nd4j reduce
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1024, 768);
        INDArray mean = input.mean(2);  // reduce along last axis
        log.info("Direct reduce: input=[1,1024,768] → mean shape={}", Arrays.toString(mean.shape()));

        // SameDiff reduce
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 1024, 768);
        SDVariable reducedMean = sd.mean("mean_out", x, false, -1);
        sd.setOutputs(Collections.singletonList("mean_out"));

        Map<String, INDArray> result = sd.output(Map.of("x", input), "mean_out");
        log.info("SameDiff reduce: output shape={}", Arrays.toString(result.get("mean_out").shape()));

        // Verify values match
        INDArray diff = mean.sub(result.get("mean_out"));
        double maxDiff = diff.amaxNumber().doubleValue();
        log.info("Max difference between direct and SameDiff: {}", maxDiff);

        input.close();
        mean.close();
        log.info("=== Test passed ===");
    }

    @Test
    public void testRepeatedVLMChunkPipeline() {
        log.info("=== Test: Repeated VLM-like chunk pipeline with many ops ===");
        log.info("This reproduces the VLM crash: many iterations × many ops per iteration");

        // Build a larger SameDiff graph mimicking the VLM vision encoder
        // The actual VLM has ~1600 ops per chunk. We simulate ~200 to stress-test.
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1024, 768);

        // Many constants (VLM has ~3000 constants for position embeddings)
        SDVariable[] constants = new SDVariable[50];
        for (int i = 0; i < 50; i++) {
            constants[i] = sd.constant("const_" + i, Nd4j.randn(DataType.FLOAT, 1, 1024, 768).mul(0.01));
        }

        // Chain of adds (simulating residual connections)
        SDVariable current = input;
        for (int i = 0; i < 50; i++) {
            current = current.add("add_" + i, constants[i]);
        }

        // ReduceMean along last axis (the fixed crash point)
        SDVariable mean = sd.mean("reduce_mean", current, false, -1);  // [batch, 1024]

        // More processing: reshape, matmul-like ops
        SDVariable reshaped = sd.reshape("reshape_1", mean, 1, 32, 32);
        SDVariable mean2 = sd.mean("reduce_mean_2", reshaped, false, -1);  // [1, 32]
        SDVariable sum = sd.sum("reduce_sum", mean2, false, -1);  // [1] or scalar

        // Boolean chain
        SDVariable zeroScalar = sd.constant("zero_scalar", Nd4j.scalar(DataType.FLOAT, 0.0f));
        SDVariable eq = sd.eq("equals", sum, zeroScalar);
        SDVariable falseConst = sd.constant("false_const", Nd4j.createFromArray(new boolean[]{false}));
        SDVariable notEquals = sd.neq("not_equals", eq, falseConst);

        sd.setOutputs(Collections.singletonList("not_equals"));

        // Run many iterations
        for (int i = 0; i < 10; i++) {
            INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 1024, 768);
            log.info("Iteration {}: starting", i);
            Map<String, INDArray> result = sd.output(Map.of("input", inputArr), "not_equals");
            INDArray output = result.get("not_equals");
            log.info("Iteration {}: output shape={}, dtype={}", i,
                    Arrays.toString(output.shape()), output.dataType());
            inputArr.close();
        }
        log.info("=== Test passed: all 10 iterations completed ===");
    }
}
