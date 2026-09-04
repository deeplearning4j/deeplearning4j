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

import static org.junit.jupiter.api.Assertions.*;

import java.io.File;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.*;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.nativeblas.OpaqueNDArray;

@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class SameDiffConcurrencyTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @BeforeEach
    public void before() {
        Nd4j.create(1);
        Nd4j.getRandom().setSeed(123);
    }

    @AfterEach
    public void after() {
        Nd4j.getNativeOps().enableDebugMode(false);
        Nd4j.getNativeOps().enableVerboseMode(false);
    }

    /**
     * Multi-threaded test simulating Spring server concurrent requests.
     * Multiple threads share a single SameDiff model instance (like Spring singleton)
     * and execute the cast operation concurrently.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCastLongToIntMultiThreadedSharedModel(Nd4jBackend backend) throws Exception {
        int nThreads = 8;
        int nIterationsPerThread = 50;
        int seqLength = 512;

        SameDiff sd = SameDiff.create();
        SDVariable inputIds = sd.placeHolder("input_ids", DataType.LONG, -1, -1);
        SDVariable inputIdsInt32 = inputIds.castTo("input_ids_int32", DataType.INT);

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(nThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        AtomicReference<Throwable> firstError = new AtomicReference<>();

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            Thread thread = new Thread(() -> {
                try {
                    startLatch.await();

                    for (int iter = 0; iter < nIterationsPerThread; iter++) {
                        int batchSize = (threadId % 4) + 1;

                        INDArray input = Nd4j.zeros(DataType.LONG, batchSize, seqLength);
                        for (int b = 0; b < batchSize; b++) {
                            input.putScalar(b, 0, 101);
                            input.putScalar(b, 1, 2000 + threadId * 100 + iter);
                            input.putScalar(b, 2, 102);
                        }

                        Map<String, INDArray> placeholders = Collections.singletonMap("input_ids", input);
                        Map<String, INDArray> outputs = sd.output(placeholders, "input_ids_int32");

                        INDArray result = outputs.get("input_ids_int32");
                        if (result == null) {
                            throw new RuntimeException("Null result at thread " + threadId + " iter " + iter);
                        }
                        if (result.dataType() != DataType.INT) {
                            throw new RuntimeException("Wrong dtype at thread " + threadId + " iter " + iter);
                        }
                        if (result.shape()[0] != batchSize || result.shape()[1] != seqLength) {
                            throw new RuntimeException("Wrong shape at thread " + threadId + " iter " + iter);
                        }
                        if (result.getInt(0, 0) != 101) {
                            throw new RuntimeException("Wrong value at thread " + threadId + " iter " + iter);
                        }

                        result.close();
                        input.close();
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    firstError.compareAndSet(null, e);
                    log.error("Thread {} failed", threadId, e);
                } finally {
                    doneLatch.countDown();
                }
            });
            thread.setName("CastTest-Thread-" + threadId);
            thread.start();
        }

        startLatch.countDown();
        doneLatch.await();

        if (failed.get()) {
            Throwable error = firstError.get();
            if (error != null) {
                throw new RuntimeException("Multi-threaded test failed", error);
            }
            fail("Multi-threaded test failed with unknown error");
        }
    }

    /**
     * Test concurrent model creation and execution.
     * Simulates Spring startup where multiple beans might initialize encoders concurrently.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCastLongToIntConcurrentModelCreation(Nd4jBackend backend) throws Exception {
        int nThreads = 4;
        int seqLength = 512;

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(nThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        AtomicReference<Throwable> firstError = new AtomicReference<>();

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            Thread thread = new Thread(() -> {
                try {
                    startLatch.await();

                    SameDiff sd = SameDiff.create();
                    SDVariable inputIds = sd.placeHolder("input_ids", DataType.LONG, -1, -1);
                    SDVariable inputIdsInt32 = inputIds.castTo("input_ids_int32", DataType.INT);

                    for (int iter = 0; iter < 20; iter++) {
                        int batchSize = (iter % 4) + 1;
                        INDArray input = Nd4j.zeros(DataType.LONG, batchSize, seqLength);
                        input.putScalar(0, 0, 101 + threadId);

                        Map<String, INDArray> outputs = sd.output(
                            Collections.singletonMap("input_ids", input), "input_ids_int32");

                        INDArray result = outputs.get("input_ids_int32");
                        assertEquals(DataType.INT, result.dataType());
                        assertEquals(101 + threadId, result.getInt(0, 0));

                        result.close();
                        input.close();
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    firstError.compareAndSet(null, e);
                    log.error("Thread {} failed", threadId, e);
                } finally {
                    doneLatch.countDown();
                }
            });
            thread.setName("ModelCreate-Thread-" + threadId);
            thread.start();
        }

        startLatch.countDown();
        doneLatch.await();

        if (failed.get()) {
            throw new RuntimeException("Concurrent model creation test failed", firstError.get());
        }
    }

    /**
     * Test simulating Spring request handling with model loaded from file.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCastLongToIntLoadedModelMultiThreaded(Nd4jBackend backend) throws Exception {
        int nThreads = 8;
        int nIterationsPerThread = 25;
        int seqLength = 512;

        SameDiff original = SameDiff.create();
        SDVariable inputIds = original.placeHolder("input_ids", DataType.LONG, -1, -1);
        SDVariable attentionMask = original.placeHolder("attention_mask", DataType.LONG, -1, -1);
        SDVariable inputIdsInt32 = inputIds.castTo("input_ids_int32", DataType.INT);
        SDVariable maskInt32 = attentionMask.castTo("attention_mask_int32", DataType.INT);

        File tempFile = File.createTempFile("mt_cast_model", ".fb");
        tempFile.deleteOnExit();
        original.asFlatFile(tempFile);

        SameDiff loaded = SameDiff.fromFlatFile(tempFile);

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(nThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        AtomicReference<Throwable> firstError = new AtomicReference<>();

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            Thread thread = new Thread(() -> {
                try {
                    startLatch.await();

                    for (int iter = 0; iter < nIterationsPerThread; iter++) {
                        int batchSize = (threadId % 4) + 1;

                        INDArray inputData = Nd4j.zeros(DataType.LONG, batchSize, seqLength);
                        INDArray maskData = Nd4j.ones(DataType.LONG, batchSize, seqLength);

                        for (int b = 0; b < batchSize; b++) {
                            inputData.putScalar(b, 0, 101);
                            inputData.putScalar(b, 1, 2000 + threadId);
                            inputData.putScalar(b, 2, 102);
                        }

                        Map<String, INDArray> placeholders = new HashMap<>();
                        placeholders.put("input_ids", inputData);
                        placeholders.put("attention_mask", maskData);

                        Map<String, INDArray> outputs = loaded.output(
                            placeholders, "input_ids_int32", "attention_mask_int32");

                        assertNotNull(outputs.get("input_ids_int32"));
                        assertNotNull(outputs.get("attention_mask_int32"));
                        assertEquals(DataType.INT, outputs.get("input_ids_int32").dataType());

                        outputs.get("input_ids_int32").close();
                        outputs.get("attention_mask_int32").close();
                        inputData.close();
                        maskData.close();
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    firstError.compareAndSet(null, e);
                    log.error("Thread {} failed", threadId, e);
                } finally {
                    doneLatch.countDown();
                }
            });
            thread.setName("LoadedModel-Thread-" + threadId);
            thread.start();
        }

        startLatch.countDown();
        doneLatch.await();

        if (failed.get()) {
            throw new RuntimeException("Loaded model multi-threaded test failed", firstError.get());
        }
    }

    /**
     * Rapid-fire test with quick successive executions to stress buffer allocation.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCastLongToIntRapidFireExecution(Nd4jBackend backend) throws Exception {
        int nThreads = 4;
        int nIterationsPerThread = 200;
        int seqLength = 512;

        SameDiff sd = SameDiff.create();
        SDVariable inputIds = sd.placeHolder("input_ids", DataType.LONG, -1, -1);
        SDVariable inputIdsInt32 = inputIds.castTo("input_ids_int32", DataType.INT);

        ExecutorService executor = Executors.newFixedThreadPool(nThreads);
        AtomicInteger successCount = new AtomicInteger(0);
        AtomicBoolean failed = new AtomicBoolean(false);

        List<Future<?>> futures = new ArrayList<>();

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            futures.add(executor.submit(() -> {
                try {
                    for (int iter = 0; iter < nIterationsPerThread; iter++) {
                        INDArray input = Nd4j.zeros(DataType.LONG, 1, seqLength);
                        input.putScalar(0, 0, 101);

                        Map<String, INDArray> outputs = sd.output(
                            Collections.singletonMap("input_ids", input), "input_ids_int32");

                        INDArray result = outputs.get("input_ids_int32");
                        if (result == null || result.dataType() != DataType.INT) {
                            failed.set(true);
                            return;
                        }

                        successCount.incrementAndGet();
                        result.close();
                        input.close();
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    log.error("Rapid fire thread {} failed", threadId, e);
                }
            }));
        }

        for (Future<?> f : futures) {
            f.get();
        }
        executor.shutdown();

        assertFalse(failed.get(), "Rapid fire test had failures");
        assertEquals(nThreads * nIterationsPerThread, successCount.get(),
            "Expected all iterations to succeed");
    }

    /**
     * Test concurrent model LOADING to reproduce heap corruption during deserialization.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcurrentModelLoadingWithScalars(Nd4jBackend backend) throws Exception {
        int nThreads = 8;
        int nLoadsPerThread = 10;

        SameDiff original = SameDiff.create();

        SDVariable input = original.placeHolder("input", DataType.FLOAT, -1, 128);

        SDVariable scale = original.constant("scale", Nd4j.scalar(DataType.FLOAT, 0.125f));
        SDVariable bias = original.constant("bias", Nd4j.scalar(DataType.FLOAT, 0.0f));
        SDVariable epsilon = original.constant("epsilon", Nd4j.scalar(DataType.FLOAT, 1e-6f));

        SDVariable weight1 = original.var("weight1", Nd4j.randn(DataType.FLOAT, 128, 64));
        SDVariable weight2 = original.var("weight2", Nd4j.randn(DataType.FLOAT, 64, 32));

        SDVariable layer1 = original.mmul("layer1", input, weight1);
        SDVariable scaled = layer1.mul("scaled", scale);
        SDVariable biased = scaled.add("biased", bias);
        SDVariable layer2 = original.mmul("layer2", biased, weight2);
        SDVariable output = original.nn.softmax("output", layer2, -1);

        File tempFile = File.createTempFile("concurrent_load_test", ".fb");
        tempFile.deleteOnExit();
        original.asFlatFile(tempFile);

        log.info("Created model with scalars, saving to: {}", tempFile.getAbsolutePath());

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(nThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        AtomicReference<Throwable> firstError = new AtomicReference<>();
        AtomicInteger successCount = new AtomicInteger(0);

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            Thread thread = new Thread(() -> {
                try {
                    startLatch.await();

                    for (int i = 0; i < nLoadsPerThread; i++) {
                        SameDiff loaded = SameDiff.fromFlatFile(tempFile);

                        assertNotNull(loaded, "Loaded model should not be null");
                        assertNotNull(loaded.getVariable("scale"), "Scale constant should exist");
                        assertNotNull(loaded.getVariable("bias"), "Bias constant should exist");
                        assertNotNull(loaded.getVariable("epsilon"), "Epsilon constant should exist");

                        INDArray inputData = Nd4j.randn(DataType.FLOAT, 2, 128);

                        Map<String, INDArray> placeholders = new HashMap<>();
                        placeholders.put("input", inputData);

                        Map<String, INDArray> outputs = loaded.output(placeholders, "output");
                        assertNotNull(outputs.get("output"), "Output should not be null");

                        outputs.get("output").close();
                        inputData.close();

                        successCount.incrementAndGet();

                        if (i % 5 == 0) {
                            log.info("Thread {} completed load iteration {}/{}", threadId, i + 1, nLoadsPerThread);
                        }
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    firstError.compareAndSet(null, e);
                    log.error("Thread {} failed during model loading", threadId, e);
                } finally {
                    doneLatch.countDown();
                }
            });
            thread.setName("ModelLoader-Thread-" + threadId);
            thread.start();
        }

        startLatch.countDown();

        boolean completed = doneLatch.await(120, TimeUnit.SECONDS);
        assertTrue(completed, "All threads should complete within timeout");

        if (failed.get()) {
            throw new RuntimeException("Concurrent model loading test failed", firstError.get());
        }

        assertEquals(nThreads * nLoadsPerThread, successCount.get(),
            "All model loads should succeed");

        log.info("Concurrent model loading test passed: {} successful loads", successCount.get());
    }

    /**
     * Test concurrent model loading with SDZ format (compressed).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcurrentModelLoadingSDZ(Nd4jBackend backend) throws Exception {
        int nThreads = 8;
        int nLoadsPerThread = 5;

        SameDiff original = SameDiff.create();

        SDVariable input = original.placeHolder("input", DataType.FLOAT, -1, 768);

        SDVariable dim0 = original.constant("dim_0", Nd4j.scalar(DataType.LONG, 0L));
        SDVariable dim1 = original.constant("dim_1", Nd4j.scalar(DataType.LONG, 1L));
        SDVariable dimNeg1 = original.constant("dim_neg1", Nd4j.scalar(DataType.LONG, -1L));

        SDVariable lnGamma = original.var("ln_gamma", Nd4j.ones(DataType.FLOAT, 768));
        SDVariable lnBeta = original.var("ln_beta", Nd4j.zeros(DataType.FLOAT, 768));
        SDVariable lnEps = original.constant("ln_eps", Nd4j.scalar(DataType.FLOAT, 1e-12f));

        SDVariable denseWeight = original.var("dense_weight", Nd4j.randn(DataType.FLOAT, 768, 768).mul(0.02));
        SDVariable denseBias = original.var("dense_bias", Nd4j.zeros(DataType.FLOAT, 768));

        SDVariable matmulDim0 = original.constant("matmul_dim0", Nd4j.scalar(DataType.INT, 0));
        SDVariable matmulDim1 = original.constant("matmul_dim1", Nd4j.scalar(DataType.INT, 1));

        SDVariable dense = original.nn.linear("dense", input, denseWeight, denseBias);
        SDVariable normalized = original.nn.layerNorm("output", dense, lnGamma, lnBeta, true, 1);

        File tempFile = File.createTempFile("concurrent_load_sdz_test", ".sdz");
        tempFile.deleteOnExit();
        original.save(tempFile, true);

        log.info("Created SDZ model with scalars, saving to: {}", tempFile.getAbsolutePath());

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(nThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        AtomicReference<Throwable> firstError = new AtomicReference<>();
        AtomicInteger successCount = new AtomicInteger(0);

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            Thread thread = new Thread(() -> {
                try {
                    startLatch.await();

                    for (int i = 0; i < nLoadsPerThread; i++) {
                        SameDiff loaded = SameDiff.load(tempFile, true);

                        assertNotNull(loaded.getVariable("dim_0"), "dim_0 should exist");
                        assertNotNull(loaded.getVariable("dim_1"), "dim_1 should exist");
                        assertNotNull(loaded.getVariable("ln_eps"), "ln_eps should exist");
                        assertNotNull(loaded.getVariable("matmul_dim0"), "matmul_dim0 should exist");
                        assertNotNull(loaded.getVariable("matmul_dim1"), "matmul_dim1 should exist");

                        INDArray inputData = Nd4j.randn(DataType.FLOAT, 2, 768);
                        Map<String, INDArray> placeholders = new HashMap<>();
                        placeholders.put("input", inputData);

                        Map<String, INDArray> outputs = loaded.output(placeholders, "output");
                        assertNotNull(outputs.get("output"));
                        assertEquals(DataType.FLOAT, outputs.get("output").dataType());

                        outputs.get("output").close();
                        inputData.close();

                        successCount.incrementAndGet();
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    firstError.compareAndSet(null, e);
                    log.error("Thread {} failed during SDZ model loading", threadId, e);
                } finally {
                    doneLatch.countDown();
                }
            });
            thread.setName("SDZLoader-Thread-" + threadId);
            thread.start();
        }

        startLatch.countDown();
        boolean completed = doneLatch.await(120, TimeUnit.SECONDS);
        assertTrue(completed, "All threads should complete within timeout");

        if (failed.get()) {
            throw new RuntimeException("Concurrent SDZ model loading test failed", firstError.get());
        }

        assertEquals(nThreads * nLoadsPerThread, successCount.get());
        log.info("Concurrent SDZ model loading test passed: {} successful loads", successCount.get());
    }

    /**
     * Stress test for concurrent array allocation and deallocation.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcurrentAllocationDeallocationStress(Nd4jBackend backend) throws Exception {
        int nThreads = 16;
        int nIterationsPerThread = 100;
        int arraysPerIteration = 10;

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(nThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        AtomicReference<Throwable> firstError = new AtomicReference<>();
        AtomicInteger successCount = new AtomicInteger(0);

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            Thread thread = new Thread(() -> {
                try {
                    startLatch.await();

                    for (int i = 0; i < nIterationsPerThread; i++) {
                        List<INDArray> arrays = new ArrayList<>();
                        for (int j = 0; j < arraysPerIteration; j++) {
                            int size = 64 + (j * 32);
                            INDArray arr = Nd4j.randn(DataType.FLOAT, size, size);
                            INDArray result = arr.mul(2.0).add(1.0);
                            arrays.add(result);
                            arr.close();
                        }

                        for (int j = 0; j < arraysPerIteration / 2; j++) {
                            arrays.get(j).close();
                        }

                        for (int j = arraysPerIteration / 2; j < arraysPerIteration; j++) {
                            arrays.get(j).close();
                        }
                        arrays.clear();

                        if (i % 10 == 0) {
                            Nd4j.getMemoryManager().invokeGc();
                        }

                        successCount.incrementAndGet();
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    firstError.compareAndSet(null, e);
                    log.error("Thread {} failed during allocation/deallocation stress", threadId, e);
                } finally {
                    doneLatch.countDown();
                }
            });
            thread.setName("AllocDealloc-Thread-" + threadId);
            thread.start();
        }

        startLatch.countDown();
        boolean completed = doneLatch.await(180, TimeUnit.SECONDS);
        assertTrue(completed, "All threads should complete within timeout");

        if (failed.get()) {
            throw new RuntimeException("Concurrent allocation/deallocation stress test failed", firstError.get());
        }

        assertEquals(nThreads * nIterationsPerThread, successCount.get());
        log.info("Concurrent allocation/deallocation stress test passed: {} successful iterations", successCount.get());
    }

    /**
     * Test concurrent model loading with explicit deallocation.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcurrentModelLoadingWithExplicitDeallocation(Nd4jBackend backend) throws Exception {
        int nThreads = 8;
        int nLoadsPerThread = 10;

        SameDiff original = SameDiff.create();
        SDVariable input = original.placeHolder("input", DataType.FLOAT, -1, 256);

        SDVariable w1 = original.var("w1", Nd4j.randn(DataType.FLOAT, 256, 512).mul(0.01));
        SDVariable b1 = original.var("b1", Nd4j.zeros(DataType.FLOAT, 512));
        SDVariable w2 = original.var("w2", Nd4j.randn(DataType.FLOAT, 512, 256).mul(0.01));
        SDVariable b2 = original.var("b2", Nd4j.zeros(DataType.FLOAT, 256));
        SDVariable w3 = original.var("w3", Nd4j.randn(DataType.FLOAT, 256, 128).mul(0.01));
        SDVariable b3 = original.var("b3", Nd4j.zeros(DataType.FLOAT, 128));

        for (int i = 0; i < 20; i++) {
            original.constant("scalar_int_" + i, Nd4j.scalar(DataType.INT, i));
            original.constant("scalar_long_" + i, Nd4j.scalar(DataType.LONG, (long) i));
            original.constant("scalar_float_" + i, Nd4j.scalar(DataType.FLOAT, (float) i * 0.1f));
        }

        SDVariable h1 = original.nn.relu(input.mmul(w1).add(b1), 0);
        SDVariable h2 = original.nn.relu(h1.mmul(w2).add(b2), 0);
        SDVariable output = original.nn.softmax("output", h2.mmul(w3).add(b3), -1);

        File tempFile = File.createTempFile("dealloc_stress_test", ".fb");
        tempFile.deleteOnExit();
        original.asFlatFile(tempFile);

        log.info("Created model with many arrays for deallocation stress test");

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(nThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        AtomicReference<Throwable> firstError = new AtomicReference<>();
        AtomicInteger successCount = new AtomicInteger(0);

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            Thread thread = new Thread(() -> {
                try {
                    startLatch.await();

                    for (int i = 0; i < nLoadsPerThread; i++) {
                        SameDiff loaded = SameDiff.fromFlatFile(tempFile);

                        INDArray inputData = Nd4j.randn(DataType.FLOAT, 4, 256);
                        Map<String, INDArray> placeholders = new HashMap<>();
                        placeholders.put("input", inputData);

                        Map<String, INDArray> outputs = loaded.output(placeholders, "output");
                        assertNotNull(outputs.get("output"));

                        outputs.get("output").close();
                        inputData.close();

                        if (i % 3 == 0) {
                            Nd4j.getMemoryManager().invokeGc();
                        }

                        successCount.incrementAndGet();
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    firstError.compareAndSet(null, e);
                    log.error("Thread {} failed during model loading with deallocation", threadId, e);
                } finally {
                    doneLatch.countDown();
                }
            });
            thread.setName("ModelLoader-Dealloc-Thread-" + threadId);
            thread.start();
        }

        startLatch.countDown();

        boolean completed = doneLatch.await(120, TimeUnit.SECONDS);
        assertTrue(completed, "All threads should complete within timeout");

        if (failed.get()) {
            throw new RuntimeException("Concurrent model loading with deallocation test failed", firstError.get());
        }

        assertEquals(nThreads * nLoadsPerThread, successCount.get());
        log.info("Concurrent model loading with deallocation test passed: {} successful loads", successCount.get());
    }

    /**
     * Test rapid creation and destruction of SameDiff instances.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRapidSameDiffCreationDestruction(Nd4jBackend backend) throws Exception {
        int nThreads = 12;
        int nModelsPerThread = 50;

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(nThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        AtomicReference<Throwable> firstError = new AtomicReference<>();
        AtomicInteger successCount = new AtomicInteger(0);

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            Thread thread = new Thread(() -> {
                try {
                    startLatch.await();

                    for (int i = 0; i < nModelsPerThread; i++) {
                        SameDiff sd = SameDiff.create();
                        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 64);
                        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 64, 32).mul(0.1));
                        SDVariable b = sd.var("b", Nd4j.zeros(DataType.FLOAT, 32));

                        sd.constant("s1", Nd4j.scalar(DataType.INT, threadId));
                        sd.constant("s2", Nd4j.scalar(DataType.FLOAT, (float) i));

                        SDVariable output = sd.nn.softmax("output", input.mmul(w).add(b), -1);

                        INDArray inputData = Nd4j.randn(DataType.FLOAT, 2, 64);
                        Map<String, INDArray> outputs = sd.output(
                            Collections.singletonMap("input", inputData), "output");

                        assertNotNull(outputs.get("output"));
                        outputs.get("output").close();
                        inputData.close();

                        sd = null;

                        successCount.incrementAndGet();
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    firstError.compareAndSet(null, e);
                    log.error("Thread {} failed during rapid SameDiff creation", threadId, e);
                } finally {
                    doneLatch.countDown();
                }
            });
            thread.setName("RapidCreate-Thread-" + threadId);
            thread.start();
        }

        startLatch.countDown();
        boolean completed = doneLatch.await(180, TimeUnit.SECONDS);
        assertTrue(completed, "All threads should complete within timeout");

        if (failed.get()) {
            throw new RuntimeException("Rapid SameDiff creation/destruction test failed", firstError.get());
        }

        assertEquals(nThreads * nModelsPerThread, successCount.get());
        log.info("Rapid SameDiff creation/destruction test passed: {} successful models", successCount.get());
    }

    /**
     * Test concurrent scalar constant creation.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcurrentScalarConstantCreation(Nd4jBackend backend) throws Exception {
        int nThreads = 16;
        int nScalarsPerThread = 200;

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(nThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        AtomicReference<Throwable> firstError = new AtomicReference<>();
        AtomicInteger successCount = new AtomicInteger(0);

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            Thread thread = new Thread(() -> {
                try {
                    startLatch.await();

                    for (int i = 0; i < nScalarsPerThread; i++) {
                        INDArray intScalar = Nd4j.scalar(DataType.INT, threadId * 1000 + i);
                        INDArray longScalar = Nd4j.scalar(DataType.LONG, (long) threadId * 1000 + i);
                        INDArray floatScalar = Nd4j.scalar(DataType.FLOAT, (float) i * 0.001f);
                        INDArray doubleScalar = Nd4j.scalar(DataType.DOUBLE, (double) i * 0.001);

                        INDArray result = floatScalar.mul(2.0).add(doubleScalar);

                        assertEquals(1, intScalar.length());
                        assertEquals(1, longScalar.length());
                        assertEquals(1, floatScalar.length());
                        assertEquals(1, result.length());

                        intScalar.close();
                        longScalar.close();

                        successCount.incrementAndGet();
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    firstError.compareAndSet(null, e);
                    log.error("Thread {} failed during scalar creation", threadId, e);
                } finally {
                    doneLatch.countDown();
                }
            });
            thread.setName("ScalarCreate-Thread-" + threadId);
            thread.start();
        }

        startLatch.countDown();
        boolean completed = doneLatch.await(120, TimeUnit.SECONDS);

        System.gc();
        Thread.sleep(500);
        System.gc();

        assertTrue(completed, "All threads should complete within timeout");

        if (failed.get()) {
            throw new RuntimeException("Concurrent scalar constant creation test failed", firstError.get());
        }

        assertEquals(nThreads * nScalarsPerThread, successCount.get());
        log.info("Concurrent scalar constant creation test passed: {} successful scalar creations", successCount.get());
    }

    /**
     * Test concurrent OpaqueNDArray creation and destruction.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcurrentOpaqueNDArrayDeallocator(Nd4jBackend backend) throws Exception {
        int nThreads = 12;
        int nIterationsPerThread = 50;

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(nThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        AtomicReference<Throwable> firstError = new AtomicReference<>();
        AtomicInteger successCount = new AtomicInteger(0);

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            Thread thread = new Thread(() -> {
                try {
                    startLatch.await();

                    for (int i = 0; i < nIterationsPerThread; i++) {
                        INDArray arr1 = Nd4j.randn(DataType.FLOAT, 32 + threadId, 64);
                        INDArray arr2 = Nd4j.randn(DataType.FLOAT, 64, 32 + threadId);
                        INDArray arr3 = Nd4j.zeros(DataType.FLOAT, 16, 16);

                        OpaqueNDArray opaque1 = OpaqueNDArray.fromINDArray(arr1);
                        OpaqueNDArray opaque2 = OpaqueNDArray.fromINDArray(arr2);
                        OpaqueNDArray opaque3 = OpaqueNDArray.fromINDArray(arr3);

                        assertFalse(opaque1.isNull());
                        assertFalse(opaque2.isNull());
                        assertFalse(opaque3.isNull());

                        opaque1.close();
                        opaque2.close();
                        opaque3.close();
                        arr1.close();
                        arr2.close();
                        arr3.close();

                        if (i % 10 == 0) {
                            Nd4j.getMemoryManager().invokeGc();
                        }

                        successCount.incrementAndGet();
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    firstError.compareAndSet(null, e);
                    log.error("Thread {} failed during OpaqueNDArray test", threadId, e);
                } finally {
                    doneLatch.countDown();
                }
            });
            thread.setName("OpaqueNDArray-Thread-" + threadId);
            thread.start();
        }

        startLatch.countDown();
        boolean completed = doneLatch.await(120, TimeUnit.SECONDS);
        assertTrue(completed, "All threads should complete within timeout");

        if (failed.get()) {
            throw new RuntimeException("Concurrent OpaqueNDArray deallocator test failed", firstError.get());
        }

        assertEquals(nThreads * nIterationsPerThread, successCount.get());
        log.info("Concurrent OpaqueNDArray deallocator test passed: {} successful iterations", successCount.get());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOpaqueNDArrayPhantomCleanup(Nd4jBackend backend) throws Exception {
        DeallocatorService service = Nd4j.getDeallocatorService();

        try (INDArray parent = Nd4j.ones(DataType.FLOAT, 32, 32)) {
            OpaqueNDArray explicitlyClosed = OpaqueNDArray.fromINDArray(parent);
            long explicitId = explicitlyClosed.getDeallocator().getUniqueId();
            OpaqueNDArray cleanupFacade = explicitlyClosed.getDeallocator().getArray();
            assertNotSame(explicitlyClosed, cleanupFacade);
            assertNotEquals(explicitlyClosed.address(), cleanupFacade.address());
            assertEquals(parent.length(), cleanupFacade.length());
            assertTrue(service.getReferenceMap().containsKey(explicitId));
            service.flushCollectedReferences();
            assertTrue(service.getReferenceMap().containsKey(explicitId),
                    "Collected-only flush must preserve a live array registration");
            explicitlyClosed.close();
            assertTrue(explicitlyClosed.isNull());
            assertFalse(service.getReferenceMap().containsKey(explicitId),
                    "Explicit close must retire its array phantom registration immediately");
        }

        AtomicLong phantomId = new AtomicLong(-1L);
        Thread allocator = new Thread(() -> {
            INDArray parent = Nd4j.ones(DataType.FLOAT, 32, 32);
            OpaqueNDArray unclosed = OpaqueNDArray.fromINDArray(parent);
            phantomId.set(unclosed.getDeallocator().getUniqueId());
        }, "OpaqueNDArray-Phantom-Allocator");
        allocator.start();
        allocator.join();

        for (int attempt = 0;
             attempt < 40 && service.getReferenceMap().containsKey(phantomId.get());
             attempt++) {
            System.gc();
            Thread.sleep(25L);
            service.flushCollectedReferences();
        }
        assertFalse(service.getReferenceMap().containsKey(phantomId.get()),
                "GC-only OpaqueNDArray cleanup must not be retained by its cleanup action");
    }

    /**
     * Test concurrent OpaqueNDArrayArr creation and destruction.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcurrentOpaqueNDArrayArrDeallocator(Nd4jBackend backend) throws Exception {
        int nThreads = 10;
        int nIterationsPerThread = 30;

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(nThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        AtomicReference<Throwable> firstError = new AtomicReference<>();
        AtomicInteger successCount = new AtomicInteger(0);

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            Thread thread = new Thread(() -> {
                try {
                    startLatch.await();

                    for (int i = 0; i < nIterationsPerThread; i++) {
                        INDArray[] arrays = new INDArray[5];
                        for (int j = 0; j < arrays.length; j++) {
                            arrays[j] = Nd4j.randn(DataType.FLOAT, 32 + j, 64 + threadId);
                        }

                        org.nd4j.nativeblas.OpaqueNDArrayArr opaqueArr =
                            org.nd4j.nativeblas.OpaqueNDArrayArr.createFrom(arrays);

                        assertNotNull(opaqueArr);
                        assertEquals(arrays.length, opaqueArr.getNumArrays());

                        opaqueArr.close();

                        for (INDArray arr : arrays) {
                            arr.close();
                        }

                        if (i % 5 == 0) {
                            Nd4j.getMemoryManager().invokeGc();
                        }

                        successCount.incrementAndGet();
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    firstError.compareAndSet(null, e);
                    log.error("Thread {} failed during OpaqueNDArrayArr test", threadId, e);
                } finally {
                    doneLatch.countDown();
                }
            });
            thread.setName("OpaqueNDArrayArr-Thread-" + threadId);
            thread.start();
        }

        startLatch.countDown();
        boolean completed = doneLatch.await(120, TimeUnit.SECONDS);
        assertTrue(completed, "All threads should complete within timeout");

        if (failed.get()) {
            throw new RuntimeException("Concurrent OpaqueNDArrayArr deallocator test failed", firstError.get());
        }

        assertEquals(nThreads * nIterationsPerThread, successCount.get());
        log.info("Concurrent OpaqueNDArrayArr deallocator test passed: {} successful iterations", successCount.get());
    }

    /**
     * Test concurrent OpaqueDataBuffer creation and destruction.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcurrentOpaqueDataBufferDeallocator(Nd4jBackend backend) throws Exception {
        int nThreads = 12;
        int nIterationsPerThread = 50;

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(nThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        AtomicReference<Throwable> firstError = new AtomicReference<>();
        AtomicInteger successCount = new AtomicInteger(0);

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            Thread thread = new Thread(() -> {
                try {
                    startLatch.await();

                    for (int i = 0; i < nIterationsPerThread; i++) {
                        OpaqueDataBuffer buffer1 =
                            OpaqueDataBuffer.allocateDataBuffer(
                                1024 + threadId * 10, DataType.FLOAT, true);
                        OpaqueDataBuffer buffer2 =
                            OpaqueDataBuffer.allocateDataBuffer(
                                512 + threadId * 5, DataType.DOUBLE, true);
                        OpaqueDataBuffer buffer3 =
                            OpaqueDataBuffer.allocateDataBuffer(
                                256 + i, DataType.INT, true);

                        assertFalse(buffer1.isNull());
                        assertFalse(buffer2.isNull());
                        assertFalse(buffer3.isNull());

                        buffer1.closeBuffer();
                        buffer2.closeBuffer();
                        buffer3.closeBuffer();

                        if (i % 10 == 0) {
                            Nd4j.getMemoryManager().invokeGc();
                        }

                        successCount.incrementAndGet();
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    firstError.compareAndSet(null, e);
                    log.error("Thread {} failed during OpaqueDataBuffer test", threadId, e);
                } finally {
                    doneLatch.countDown();
                }
            });
            thread.setName("OpaqueDataBuffer-Thread-" + threadId);
            thread.start();
        }

        startLatch.countDown();
        boolean completed = doneLatch.await(120, TimeUnit.SECONDS);
        assertTrue(completed, "All threads should complete within timeout");

        if (failed.get()) {
            throw new RuntimeException("Concurrent OpaqueDataBuffer deallocator test failed", firstError.get());
        }

        assertEquals(nThreads * nIterationsPerThread, successCount.get());
        log.info("Concurrent OpaqueDataBuffer deallocator test passed: {} successful iterations", successCount.get());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOpaqueDataBufferPhantomCleanup(Nd4jBackend backend) throws Exception {
        DeallocatorService service = Nd4j.getDeallocatorService();

        OpaqueDataBuffer explicitlyClosed =
                OpaqueDataBuffer.allocateDataBuffer(4096, DataType.FLOAT, true);
        long explicitId = explicitlyClosed.getDeallocator().getUniqueId();
        OpaqueDataBuffer cleanupFacade = explicitlyClosed.getDeallocator().getBuffer();
        assertNotSame(explicitlyClosed, cleanupFacade);
        assertEquals(explicitlyClosed.address(), cleanupFacade.address());
        assertTrue(service.getReferenceMap().containsKey(explicitId));
        service.flushCollectedReferences();
        assertTrue(service.getReferenceMap().containsKey(explicitId),
                "Collected-only flush must preserve a live buffer registration");
        explicitlyClosed.closeBuffer();
        assertFalse(service.getReferenceMap().containsKey(explicitId),
                "Explicit close must retire its phantom registration immediately");

        AtomicLong phantomId = new AtomicLong(-1L);
        Thread allocator = new Thread(() -> {
            OpaqueDataBuffer unclosed =
                    OpaqueDataBuffer.allocateDataBuffer(4096, DataType.FLOAT, true);
            phantomId.set(unclosed.getDeallocator().getUniqueId());
        }, "OpaqueDataBuffer-Phantom-Allocator");
        allocator.start();
        allocator.join();

        for (int attempt = 0;
             attempt < 40 && service.getReferenceMap().containsKey(phantomId.get());
             attempt++) {
            System.gc();
            Thread.sleep(25L);
            service.flushCollectedReferences();
        }
        assertFalse(service.getReferenceMap().containsKey(phantomId.get()),
                "GC-only OpaqueDataBuffer cleanup must not be retained by its cleanup action");
    }

    /**
     * Test concurrent deallocation race conditions by NOT explicitly closing resources,
     * forcing cleanup through GC and the DeallocatorService.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcurrentGCDeallocatorRace(Nd4jBackend backend) throws Exception {
        int nThreads = 8;
        int nIterationsPerThread = 30;

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(nThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        AtomicReference<Throwable> firstError = new AtomicReference<>();
        AtomicInteger successCount = new AtomicInteger(0);

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            Thread thread = new Thread(() -> {
                try {
                    startLatch.await();

                    for (int i = 0; i < nIterationsPerThread; i++) {
                        {
                            INDArray[] arrays = new INDArray[3];
                            for (int j = 0; j < arrays.length; j++) {
                                arrays[j] = Nd4j.randn(DataType.FLOAT, 16 + j, 32 + threadId);
                            }

                            org.nd4j.nativeblas.OpaqueNDArrayArr opaqueArr =
                                org.nd4j.nativeblas.OpaqueNDArrayArr.createFrom(arrays);
                            // Let opaqueArr and arrays go out of scope
                        }

                        Nd4j.getMemoryManager().invokeGc();

                        successCount.incrementAndGet();
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    firstError.compareAndSet(null, e);
                    log.error("Thread {} failed during GC deallocator race test", threadId, e);
                } finally {
                    doneLatch.countDown();
                }
            });
            thread.setName("GCDeallocRace-Thread-" + threadId);
            thread.start();
        }

        startLatch.countDown();
        boolean completed = doneLatch.await(120, TimeUnit.SECONDS);

        Nd4j.getMemoryManager().invokeGc();
        Thread.sleep(500);
        Nd4j.getMemoryManager().invokeGc();

        assertTrue(completed, "All threads should complete within timeout");

        if (failed.get()) {
            throw new RuntimeException("Concurrent GC deallocator race test failed", firstError.get());
        }

        assertEquals(nThreads * nIterationsPerThread, successCount.get());
        log.info("Concurrent GC deallocator race test passed: {} successful iterations", successCount.get());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRepeatedModelSaveLoadMemoryStress(Nd4jBackend backend) throws Exception {
        File tempFile = new File(System.getProperty("java.io.tmpdir"), "stress_test_model.sdz");
        try {
            SameDiff sd = SameDiff.create();
            INDArray input = Nd4j.randn(DataType.FLOAT, 16, 128);
            INDArray w1 = Nd4j.randn(DataType.FLOAT, 128, 256);
            INDArray w2 = Nd4j.randn(DataType.FLOAT, 256, 128);
            INDArray w3 = Nd4j.randn(DataType.FLOAT, 128, 64);
            INDArray b1 = Nd4j.randn(DataType.FLOAT, 256);
            INDArray b2 = Nd4j.randn(DataType.FLOAT, 128);
            INDArray b3 = Nd4j.randn(DataType.FLOAT, 64);

            SDVariable sdInput = sd.placeHolder("input", DataType.FLOAT, -1, 128);
            SDVariable sdW1 = sd.var("w1", w1);
            SDVariable sdW2 = sd.var("w2", w2);
            SDVariable sdW3 = sd.var("w3", w3);
            SDVariable sdB1 = sd.var("b1", b1);
            SDVariable sdB2 = sd.var("b2", b2);
            SDVariable sdB3 = sd.var("b3", b3);

            SDVariable h1 = sd.nn().relu(sdInput.mmul(sdW1).add(sdB1), 0);
            SDVariable h2 = sd.nn().relu(h1.mmul(sdW2).add(sdB2), 0);
            SDVariable out = sd.nn().softmax("output", h2.mmul(sdW3).add(sdB3), -1);

            Map<String, INDArray> placeholders = new HashMap<>();
            placeholders.put("input", input);
            sd.output(placeholders, "output");

            sd.save(tempFile, true);

            for (int i = 0; i < 10; i++) {
                SameDiff loaded = SameDiff.load(tempFile, true);
                INDArray result = loaded.output(placeholders, "output").get("output");
                assertNotNull(result);
                assertEquals(16, result.shape()[0]);
                assertEquals(64, result.shape()[1]);

                if (i % 3 == 0) {
                    System.gc();
                    Thread.sleep(50);
                }
            }
        } finally {
            if (tempFile.exists()) {
                tempFile.delete();
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcurrentModelLoadingMemoryStress(Nd4jBackend backend) throws Exception {
        File tempFile = new File(System.getProperty("java.io.tmpdir"), "concurrent_stress_model.sdz");
        try {
            SameDiff sd = SameDiff.create();
            INDArray w = Nd4j.randn(DataType.FLOAT, 64, 64);
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 64);
            SDVariable weight = sd.var("weight", w);
            SDVariable out = sd.nn().relu("output", input.mmul(weight), 0);

            sd.save(tempFile, true);

            int numThreads = 4;
            int loadsPerThread = 5;
            Thread[] threads = new Thread[numThreads];
            Exception[] errors = new Exception[numThreads];

            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                threads[t] = new Thread(() -> {
                    try {
                        for (int i = 0; i < loadsPerThread; i++) {
                            SameDiff loaded = SameDiff.load(tempFile, true);
                            Map<String, INDArray> ph = new HashMap<>();
                            ph.put("input", Nd4j.randn(DataType.FLOAT, 8, 64));
                            INDArray result = loaded.output(ph, "output").get("output");
                            assertNotNull(result);
                        }
                    } catch (Exception e) {
                        errors[threadId] = e;
                    }
                });
                threads[t].start();
            }

            for (Thread thread : threads) {
                thread.join();
            }

            for (int t = 0; t < numThreads; t++) {
                if (errors[t] != null) {
                    throw errors[t];
                }
            }
        } finally {
            if (tempFile.exists()) {
                tempFile.delete();
            }
        }
    }

    /**
     * Test concurrent loading of the actual BGE embedding model.
     * Skipped if the model file is not present on the test system.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcurrentBgeModelLoadingAndInference(Nd4jBackend backend) throws Exception {
        File modelFile = new File(System.getProperty("bge.model.path",
                System.getProperty("user.dir") + "/bge-base-en-v1.5.sdz"));
        org.junit.jupiter.api.Assumptions.assumeTrue(modelFile.exists(),
                "BGE model not found at: " + modelFile.getAbsolutePath());

        log.info("Loading BGE model from: {}", modelFile.getAbsolutePath());
        SameDiff sharedModel = SDZSerializer.load(modelFile, true);
        log.info("Model loaded. Inputs: {}, Outputs: {}", sharedModel.inputs(), sharedModel.outputs());

        int seqLength = 512;

        java.util.function.Supplier<Map<String, INDArray>> createInputs = () -> {
            int batchSize = 1;
            Map<String, INDArray> inputMap = new HashMap<>();
            for (String inputName : sharedModel.inputs()) {
                INDArray input = Nd4j.zeros(org.nd4j.linalg.api.buffer.DataType.INT64, batchSize, seqLength);
                if (inputName.toLowerCase().contains("input_id")) {
                    input.putScalar(0, 0, 101);
                    input.putScalar(0, 1, 102);
                } else if (inputName.toLowerCase().contains("attention") || inputName.toLowerCase().contains("mask")) {
                    input.putScalar(0, 0, 1);
                    input.putScalar(0, 1, 1);
                }
                inputMap.put(inputName, input);
            }
            return inputMap;
        };

        log.info("Testing single-threaded inference first...");
        {
            Map<String, INDArray> inputMap = createInputs.get();
            try {
                Map<String, INDArray> outputs = sharedModel.output(inputMap, sharedModel.outputs());
                for (INDArray output : outputs.values()) {
                    assertNotNull(output, "Single-thread output should not be null");
                    log.info("Single-thread output shape: {}", java.util.Arrays.toString(output.shape()));
                }
                for (INDArray output : outputs.values()) {
                    output.close();
                }
                log.info("Single-threaded inference PASSED");
            } finally {
                for (INDArray input : inputMap.values()) {
                    input.close();
                }
            }
        }

        log.info("Running warm-up inferences...");
        for (int warm = 0; warm < 3; warm++) {
            Map<String, INDArray> inputMap = createInputs.get();
            try {
                Map<String, INDArray> outputs = sharedModel.output(inputMap, sharedModel.outputs());
                for (INDArray output : outputs.values()) {
                    output.close();
                }
            } finally {
                for (INDArray input : inputMap.values()) {
                    input.close();
                }
            }
        }
        log.info("Warm-up completed");

        int nThreads = 4;
        int nInferencesPerThread = 5;

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(nThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        AtomicReference<Throwable> firstError = new AtomicReference<>();
        AtomicInteger successCount = new AtomicInteger(0);

        log.info("Starting concurrent inference test with {} threads, {} inferences each", nThreads, nInferencesPerThread);

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            Thread thread = new Thread(() -> {
                try {
                    startLatch.await();

                    for (int i = 0; i < nInferencesPerThread; i++) {
                        Map<String, INDArray> inputMap = createInputs.get();
                        try {
                            Map<String, INDArray> outputs = sharedModel.output(inputMap, sharedModel.outputs());

                            for (INDArray output : outputs.values()) {
                                assertNotNull(output, "Output should not be null");
                                assertFalse(output.isNaN().any(), "Output should not contain NaN");
                            }

                            for (INDArray output : outputs.values()) {
                                output.close();
                            }

                            successCount.incrementAndGet();

                            if ((i + 1) % 2 == 0) {
                                log.info("Thread {} completed inference {}/{}", threadId, i + 1, nInferencesPerThread);
                            }
                        } finally {
                            for (INDArray input : inputMap.values()) {
                                input.close();
                            }
                        }
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    firstError.compareAndSet(null, e);
                    log.error("Thread {} failed during BGE inference at success count {}", threadId, successCount.get(), e);
                } finally {
                    doneLatch.countDown();
                }
            });
            thread.setName("BgeInference-Thread-" + threadId);
            thread.start();
        }

        startLatch.countDown();
        boolean completed = doneLatch.await(300, TimeUnit.SECONDS);

        assertTrue(completed, "All threads should complete within timeout");

        if (failed.get()) {
            throw new RuntimeException("Concurrent BGE inference test failed", firstError.get());
        }

        assertEquals(nThreads * nInferencesPerThread, successCount.get());
        log.info("Concurrent BGE inference test passed: {} successful inferences", successCount.get());
    }

    /**
     * Test that simulates loading the BGE model in multiple threads where each thread loads its own copy.
     * Skipped if the model file is not present.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcurrentBgeModelFreshLoads(Nd4jBackend backend) throws Exception {
        File modelFile = new File(System.getProperty("bge.model.path",
                System.getProperty("user.dir") + "/bge-base-en-v1.5.sdz"));
        org.junit.jupiter.api.Assumptions.assumeTrue(modelFile.exists(),
                "BGE model not found at: " + modelFile.getAbsolutePath());

        int seqLength = 512;

        log.info("Testing single load + inference first...");
        {
            SameDiff model = SDZSerializer.load(modelFile, true);
            assertNotNull(model, "Loaded model should not be null");
            log.info("Model loaded. Inputs: {}, Outputs: {}", model.inputs(), model.outputs());

            Map<String, INDArray> inputMap = new HashMap<>();
            for (String inputName : model.inputs()) {
                INDArray input = Nd4j.zeros(org.nd4j.linalg.api.buffer.DataType.INT64, 1, seqLength);
                if (inputName.toLowerCase().contains("input_id")) {
                    input.putScalar(0, 0, 101);
                    input.putScalar(0, 1, 102);
                } else if (inputName.toLowerCase().contains("attention") || inputName.toLowerCase().contains("mask")) {
                    input.putScalar(0, 0, 1);
                    input.putScalar(0, 1, 1);
                }
                inputMap.put(inputName, input);
            }

            Map<String, INDArray> outputs = model.output(inputMap, model.outputs());
            assertFalse(outputs.isEmpty(), "Should have outputs");
            log.info("Single load + inference PASSED");

            for (INDArray input : inputMap.values()) {
                input.close();
            }
            for (INDArray output : outputs.values()) {
                output.close();
            }
            model = null;
            Nd4j.getMemoryManager().invokeGc();
        }

        int nThreads = 2;
        int nLoadsPerThread = 2;

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(nThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        AtomicReference<Throwable> firstError = new AtomicReference<>();
        AtomicInteger successCount = new AtomicInteger(0);

        log.info("Starting concurrent BGE fresh loads test. {} threads x {} loads", nThreads, nLoadsPerThread);

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            final int finalSeqLength = seqLength;
            Thread thread = new Thread(() -> {
                try {
                    startLatch.await();

                    for (int i = 0; i < nLoadsPerThread; i++) {
                        log.info("Thread {} starting load {}/{}", threadId, i + 1, nLoadsPerThread);

                        SameDiff model = SDZSerializer.load(modelFile, true);
                        assertNotNull(model, "Loaded model should not be null");

                        Map<String, INDArray> inputMap = new HashMap<>();
                        for (String inputName : model.inputs()) {
                            INDArray input = Nd4j.zeros(org.nd4j.linalg.api.buffer.DataType.INT64, 1, finalSeqLength);
                            if (inputName.toLowerCase().contains("input_id")) {
                                input.putScalar(0, 0, 101);
                                input.putScalar(0, 1, 102);
                            } else if (inputName.toLowerCase().contains("attention") || inputName.toLowerCase().contains("mask")) {
                                input.putScalar(0, 0, 1);
                                input.putScalar(0, 1, 1);
                            }
                            inputMap.put(inputName, input);
                        }

                        Map<String, INDArray> outputs = model.output(inputMap, model.outputs());
                        assertFalse(outputs.isEmpty(), "Should have outputs");

                        for (INDArray input : inputMap.values()) {
                            input.close();
                        }
                        for (INDArray output : outputs.values()) {
                            output.close();
                        }

                        model = null;

                        Nd4j.getMemoryManager().invokeGc();

                        successCount.incrementAndGet();
                        log.info("Thread {} completed load {}/{}", threadId, i + 1, nLoadsPerThread);
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    firstError.compareAndSet(null, e);
                    log.error("Thread {} failed during BGE fresh load at success count {}", threadId, successCount.get(), e);
                } finally {
                    doneLatch.countDown();
                }
            });
            thread.setName("BgeFreshLoad-Thread-" + threadId);
            thread.start();
        }

        startLatch.countDown();
        boolean completed = doneLatch.await(600, TimeUnit.SECONDS);

        Nd4j.getMemoryManager().invokeGc();
        Thread.sleep(1000);
        Nd4j.getMemoryManager().invokeGc();

        assertTrue(completed, "All threads should complete within timeout");

        if (failed.get()) {
            throw new RuntimeException("Concurrent BGE fresh loads test failed", firstError.get());
        }

        assertEquals(nThreads * nLoadsPerThread, successCount.get());
        log.info("Concurrent BGE fresh loads test passed: {} successful loads", successCount.get());
    }

    /**
     * Test that simulates the "cold start" pattern seen in subprocess scenarios.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testColdStartConcurrentOps(Nd4jBackend backend) throws Exception {
        int nThreads = 8;
        int nOpsPerThread = 20;

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(nThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        AtomicReference<Throwable> firstError = new AtomicReference<>();
        AtomicInteger successCount = new AtomicInteger(0);

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            Thread thread = new Thread(() -> {
                try {
                    startLatch.await();

                    for (int i = 0; i < nOpsPerThread; i++) {
                        switch (i % 5) {
                            case 0: {
                                INDArray large = Nd4j.randn(DataType.FLOAT, 512, 512);
                                large.mul(2.0);
                                large.close();
                                break;
                            }
                            case 1: {
                                INDArray scalar = Nd4j.scalar(DataType.FLOAT, 1.0f);
                                scalar.add(scalar).close();
                                scalar.close();
                                break;
                            }
                            case 2: {
                                SameDiff sd = SameDiff.create();
                                SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 64);
                                SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 64, 32));
                                SDVariable out = sd.mmul("out", in, w);
                                INDArray input = Nd4j.randn(DataType.FLOAT, 4, 64);
                                Map<String, INDArray> ph = new HashMap<>();
                                ph.put("in", input);
                                INDArray result = sd.output(ph, "out").get("out");
                                result.close();
                                input.close();
                                break;
                            }
                            case 3: {
                                INDArray arr = Nd4j.randn(DataType.FLOAT, 128, 128);
                                arr.sum().close();
                                arr.mean().close();
                                arr.max().close();
                                arr.close();
                                break;
                            }
                            case 4: {
                                INDArray data = Nd4j.randn(DataType.FLOAT, 16, 32, 64);
                                INDArray reshaped = data.reshape(16, 32 * 64);
                                INDArray transposed = reshaped.transpose();
                                transposed.close();
                                data.close();
                                break;
                            }
                        }

                        successCount.incrementAndGet();
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    firstError.compareAndSet(null, e);
                    log.error("Thread {} failed during cold start concurrent ops", threadId, e);
                } finally {
                    doneLatch.countDown();
                }
            });
            thread.setName("ColdStart-Thread-" + threadId);
            thread.start();
        }

        startLatch.countDown();
        boolean completed = doneLatch.await(120, TimeUnit.SECONDS);

        assertTrue(completed, "All threads should complete within timeout");

        if (failed.get()) {
            throw new RuntimeException("Cold start concurrent ops test failed", firstError.get());
        }

        assertEquals(nThreads * nOpsPerThread, successCount.get());
        log.info("Cold start concurrent ops test passed: {} successful operations", successCount.get());
    }

    /**
     * Test transformer-like memory allocation pattern.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTransformerMemoryPattern(Nd4jBackend backend) throws Exception {
        int nThreads = 4;
        int nIterationsPerThread = 5;

        int batchSize = 2;
        int seqLen = 128;
        int hiddenDim = 768;
        int numHeads = 12;
        int headDim = hiddenDim / numHeads;
        int ffnDim = 3072;

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(nThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        AtomicReference<Throwable> firstError = new AtomicReference<>();
        AtomicInteger successCount = new AtomicInteger(0);

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            Thread thread = new Thread(() -> {
                try {
                    startLatch.await();

                    for (int iter = 0; iter < nIterationsPerThread; iter++) {
                        List<INDArray> tensors = new ArrayList<>();

                        tensors.add(Nd4j.randn(DataType.FLOAT, batchSize, seqLen, hiddenDim));
                        tensors.add(Nd4j.randn(DataType.FLOAT, batchSize, seqLen, 3 * hiddenDim));
                        tensors.add(Nd4j.randn(DataType.FLOAT, batchSize, numHeads, seqLen, seqLen));
                        tensors.add(Nd4j.randn(DataType.FLOAT, batchSize, seqLen, hiddenDim));
                        tensors.add(Nd4j.randn(DataType.FLOAT, batchSize, seqLen, ffnDim));
                        tensors.add(Nd4j.randn(DataType.FLOAT, batchSize, seqLen, hiddenDim));
                        tensors.add(Nd4j.randn(DataType.FLOAT, hiddenDim));
                        tensors.add(Nd4j.randn(DataType.FLOAT, hiddenDim));
                        tensors.add(Nd4j.scalar(DataType.FLOAT, 1e-6f));

                        for (int i = 0; i < tensors.size() - 1; i++) {
                            if (tensors.get(i).rank() == tensors.get(i + 1).rank() &&
                                java.util.Arrays.equals(tensors.get(i).shape(), tensors.get(i + 1).shape())) {
                                tensors.get(i).add(tensors.get(i + 1));
                            }
                        }

                        for (INDArray tensor : tensors) {
                            tensor.close();
                        }

                        successCount.incrementAndGet();

                        if (iter % 2 == 0) {
                            Nd4j.getMemoryManager().invokeGc();
                        }
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    firstError.compareAndSet(null, e);
                    log.error("Thread {} failed during transformer memory pattern test", threadId, e);
                } finally {
                    doneLatch.countDown();
                }
            });
            thread.setName("TransformerPattern-Thread-" + threadId);
            thread.start();
        }

        startLatch.countDown();
        boolean completed = doneLatch.await(180, TimeUnit.SECONDS);

        Nd4j.getMemoryManager().invokeGc();
        Thread.sleep(500);
        Nd4j.getMemoryManager().invokeGc();

        assertTrue(completed, "All threads should complete within timeout");

        if (failed.get()) {
            throw new RuntimeException("Transformer memory pattern test failed", firstError.get());
        }

        assertEquals(nThreads * nIterationsPerThread, successCount.get());
        log.info("Transformer memory pattern test passed: {} successful iterations", successCount.get());
    }
}
