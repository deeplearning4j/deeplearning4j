/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import org.nd4j.autodiff.samediff.execution.DspHandle;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests concurrent DSP plan sharing across multiple threads.
 *
 * Exercises the critical multithreading scenarios:
 * <ol>
 *   <li>Concurrent plan compilation — multiple threads trigger compile simultaneously</li>
 *   <li>Concurrent lifecycle transitions — shapes freeze while other threads execute</li>
 *   <li>Concurrent composite replay — multiple threads replay the same captured graph</li>
 *   <li>Concurrent plan cache access — multiple threads hit the same plan cache key</li>
 *   <li>Concurrent warmup→frozen transition — threads race through DSP phase gates</li>
 * </ol>
 */
@Slf4j
@Tag("dsp")
@Tag(TagNames.FULL_CI)
@DisplayName("DSP concurrent plan sharing tests")
public class DspConcurrentPlanSharingTest {

    @BeforeEach
    void setUp() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        System.setProperty("nd4j.dsp.nativeExecutor.enabled", "true");
    }

    @AfterEach
    void tearDown() {
        // Trim CUDA memory pools to return reserved memory to the driver.
        // Without this, prior tests' pool-reserved memory accumulates and
        // later tests fail with CUBLAS_STATUS_ALLOC_FAILED (error 3).
        try {
            Nd4j.getExecutioner().commit();
            System.gc();
            Thread.sleep(50);
            var nativeOps = Nd4j.getNativeOps();
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            for (int d = 0; d < numDevices; d++) {
                nativeOps.trimMemoryPool(d);
            }
        } catch (Exception e) {
            log.debug("tearDown pool trim: {}", e.getMessage());
        }
        System.clearProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED);
        System.clearProperty("nd4j.dsp.nativeExecutor.enabled");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 1: Concurrent execution of the SAME SameDiff graph
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "concurrentSharedGraph_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testConcurrentSharedGraphExecution(GraphExecutionMode mode) throws Exception {
        SameDiff sd = SameDiff.create();
        try {
            // Build graph: input -> matmul(W1) -> relu -> matmul(W2) -> output
            INDArray w1 = Nd4j.randn(DataType.FLOAT, 32, 64);
            INDArray w2 = Nd4j.randn(DataType.FLOAT, 64, 16);
            SDVariable weight1 = sd.constant("w1", w1);
            SDVariable weight2 = sd.constant("w2", w2);
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32);
            SDVariable h = sd.mmul("hidden", input, weight1);
            SDVariable activated = sd.nn().relu("relu", h, 0);
            SDVariable out = sd.mmul("output", activated, weight2);

            sd.setGraphExecutionMode(mode);

            // Pre-warm on main thread to get past initial compilation
            INDArray warmupInput = Nd4j.randn(DataType.FLOAT, 1, 32);
            for (int i = 0; i < 10; i++) {
                warmupInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
                sd.output(Collections.singletonMap("input", warmupInput), "output");
            }

            // Now launch concurrent threads
            int numThreads = 4;
            int stepsPerThread = 20;
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            AtomicInteger errors = new AtomicInteger(0);
            AtomicInteger nanErrors = new AtomicInteger(0);
            CountDownLatch startLatch = new CountDownLatch(1);
            CountDownLatch doneLatch = new CountDownLatch(numThreads);

            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    try {
                        startLatch.await();
                        INDArray threadInput = Nd4j.randn(DataType.FLOAT, 1, 32);
                        Map<String, INDArray> ph = new LinkedHashMap<>();
                        ph.put("input", threadInput);

                        for (int step = 0; step < stepsPerThread; step++) {
                            threadInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
                            try {
                                Map<String, INDArray> result = sd.output(ph, "output");
                                INDArray output = result.get("output");
                                if (output == null) {
                                    errors.incrementAndGet();
                                    log.error("[THREAD {}] step {}: null output", threadId, step);
                                } else if (output.rank() != 2) {
                                    errors.incrementAndGet();
                                    log.error("[THREAD {}] step {}: wrong rank={} shape={}",
                                            threadId, step, output.rank(), Arrays.toString(output.shape()));
                                } else if (output.isNaN().any()) {
                                    nanErrors.incrementAndGet();
                                    log.error("[THREAD {}] step {}: NaN in output", threadId, step);
                                }
                            } catch (Exception e) {
                                errors.incrementAndGet();
                                log.error("[THREAD {}] step {}: exception {}", threadId, step, e.getMessage());
                            }
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            startLatch.countDown();
            assertTrue(doneLatch.await(60, TimeUnit.SECONDS), "Threads did not complete within 60s");
            executor.shutdown();

            assertEquals(0, errors.get(), mode + ": " + errors.get() + " structural errors across " + numThreads + " threads");
            assertEquals(0, nanErrors.get(), mode + ": " + nanErrors.get() + " NaN errors across " + numThreads + " threads");
        } finally {
            sd.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 2: Concurrent compilation race — threads force compile simultaneously
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "concurrentCompileRace_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testConcurrentCompilationRace(GraphExecutionMode mode) throws Exception {
        SameDiff sd = SameDiff.create();
        try {
            // Build a graph with enough ops to make compilation non-trivial
            INDArray w1 = Nd4j.randn(DataType.FLOAT, 32, 32);
            INDArray w2 = Nd4j.randn(DataType.FLOAT, 32, 32);
            INDArray w3 = Nd4j.randn(DataType.FLOAT, 32, 16);
            SDVariable weight1 = sd.constant("w1", w1);
            SDVariable weight2 = sd.constant("w2", w2);
            SDVariable weight3 = sd.constant("w3", w3);
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32);

            SDVariable h1 = sd.mmul("h1", input, weight1);
            SDVariable a1 = sd.nn().relu("a1", h1, 0);
            SDVariable h2 = sd.mmul("h2", a1, weight2);
            SDVariable a2 = sd.nn().relu("a2", h2, 0);
            SDVariable out = sd.mmul("output", a2, weight3);

            sd.setGraphExecutionMode(mode);

            // NO pre-warmup — all threads start cold
            int numThreads = 4;
            int stepsPerThread = 15;
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            AtomicInteger errors = new AtomicInteger(0);
            CountDownLatch startLatch = new CountDownLatch(1);
            CountDownLatch doneLatch = new CountDownLatch(numThreads);

            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    try {
                        startLatch.await();
                        INDArray threadInput = Nd4j.randn(DataType.FLOAT, 1, 32);
                        Map<String, INDArray> ph = new LinkedHashMap<>();
                        ph.put("input", threadInput);

                        for (int step = 0; step < stepsPerThread; step++) {
                            threadInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
                            try {
                                Map<String, INDArray> result = sd.output(ph, "output");
                                INDArray output = result.get("output");
                                assertNotNull(output, "thread=" + threadId + " step=" + step);
                                assertEquals(2, output.rank(), "thread=" + threadId + " step=" + step + " rank");
                                assertFalse(output.isNaN().any(),
                                        "thread=" + threadId + " step=" + step + " has NaN");
                            } catch (Exception e) {
                                errors.incrementAndGet();
                                log.error("[COMPILE_RACE] thread={} step={}: {}", threadId, step, e.getMessage());
                            }
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            startLatch.countDown();
            assertTrue(doneLatch.await(60, TimeUnit.SECONDS), "Threads did not complete within 60s");
            executor.shutdown();

            assertEquals(0, errors.get(), mode + ": " + errors.get() + " errors during concurrent compilation");
        } finally {
            sd.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 3: Concurrent lifecycle phase transitions
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "concurrentLifecycleTransitions_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testConcurrentLifecycleTransitions(GraphExecutionMode mode) throws Exception {
        SameDiff sd = SameDiff.create();
        try {
            // Graph with norm ops that cause more complex lifecycle (buffer reallocation)
            INDArray w = Nd4j.randn(DataType.FLOAT, 32, 16);
            SDVariable weight = sd.constant("w", w);
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32);
            SDVariable h = sd.mmul("matmul", input, weight);
            SDVariable norm = sd.math().norm2("norm", h, 1);
            SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5));
            SDVariable maxNorm = sd.math().max("maxNorm", norm, eps);
            SDVariable out = sd.math().div("output", h, maxNorm);

            sd.setGraphExecutionMode(mode);

            // Run enough steps across threads to go through all lifecycle phases:
            // SLOT_BY_SLOT → SHAPES_FROZEN → REPLAYING
            int numThreads = 3;
            int stepsPerThread = 25;
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            AtomicInteger errors = new AtomicInteger(0);
            CountDownLatch startLatch = new CountDownLatch(1);
            CountDownLatch doneLatch = new CountDownLatch(numThreads);

            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    try {
                        startLatch.await();
                        INDArray threadInput = Nd4j.randn(DataType.FLOAT, 1, 32);
                        Map<String, INDArray> ph = new LinkedHashMap<>();
                        ph.put("input", threadInput);

                        for (int step = 0; step < stepsPerThread; step++) {
                            threadInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
                            try {
                                Map<String, INDArray> result = sd.output(ph, "output");
                                INDArray output = result.get("output");
                                assertNotNull(output, "thread=" + threadId + " step=" + step);
                                // Verify output shape is [1, 16] — the normalized matmul result
                                assertArrayEquals(new long[]{1, 16}, output.shape(),
                                        "thread=" + threadId + " step=" + step + " shape");
                                assertFalse(output.isNaN().any(),
                                        "thread=" + threadId + " step=" + step + " NaN");
                                assertFalse(output.isInfinite().any(),
                                        "thread=" + threadId + " step=" + step + " Inf");
                            } catch (Exception e) {
                                errors.incrementAndGet();
                                log.error("[LIFECYCLE] thread={} step={}: {}", threadId, step, e.getMessage());
                            }
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            startLatch.countDown();
            assertTrue(doneLatch.await(60, TimeUnit.SECONDS), "Threads did not complete within 60s");
            executor.shutdown();

            assertEquals(0, errors.get(),
                    mode + ": " + errors.get() + " errors during concurrent lifecycle transitions");
        } finally {
            sd.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 4: Independent SameDiff instances — no sharing, verify isolation
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "concurrentIndependentInstances_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testConcurrentIndependentInstances(GraphExecutionMode mode) throws Exception {
        int numThreads = 4;
        int stepsPerThread = 20;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        AtomicInteger errors = new AtomicInteger(0);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);
        for (int t = 0; t < numThreads; t++) {
            final int threadId = t;
            executor.submit(() -> {
                SameDiff sd = null;
                try {
                    startLatch.await();

                    // Each thread creates its own SameDiff + graph
                    sd = SameDiff.create();
                    INDArray w = Nd4j.randn(DataType.FLOAT, 32, 16);
                    SDVariable weight = sd.constant("w", w);
                    SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32);
                    SDVariable out = sd.mmul("output", input, weight);
                    sd.setGraphExecutionMode(mode);

                    INDArray threadInput = Nd4j.randn(DataType.FLOAT, 1, 32);
                    Map<String, INDArray> ph = new LinkedHashMap<>();
                    ph.put("input", threadInput);

                    for (int step = 0; step < stepsPerThread; step++) {
                        threadInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
                        try {
                            Map<String, INDArray> result = sd.output(ph, "output");
                            INDArray output = result.get("output");
                            assertNotNull(output, "thread=" + threadId + " step=" + step);
                            assertArrayEquals(new long[]{1, 16}, output.shape(),
                                    "thread=" + threadId + " step=" + step + " shape");
                        } catch (Exception e) {
                            errors.incrementAndGet();
                            log.error("[INDEPENDENT] thread={} step={}: {}", threadId, step, e.getMessage());
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    if (sd != null) sd.close();
                    doneLatch.countDown();
                }
            });
        }

        startLatch.countDown();
        assertTrue(doneLatch.await(60, TimeUnit.SECONDS), "Threads did not complete within 60s");
        executor.shutdown();

        assertEquals(0, errors.get(),
                mode + ": " + errors.get() + " errors in independent instances");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 5: Mixed execution modes — different threads use different modes
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("concurrent independent instances with different execution modes")
    void testConcurrentMixedModes() throws Exception {
        GraphExecutionMode[] modes = {
                GraphExecutionMode.AUTO,
                GraphExecutionMode.TRITON,
                GraphExecutionMode.CUDA_GRAPHS
        };

        int numThreads = modes.length;
        int stepsPerThread = 20;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        AtomicInteger errors = new AtomicInteger(0);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);
        for (int t = 0; t < numThreads; t++) {
            final int threadId = t;
            final GraphExecutionMode mode = modes[t];
            executor.submit(() -> {
                SameDiff sd = null;
                try {
                    startLatch.await();

                    sd = SameDiff.create();
                    INDArray w = Nd4j.randn(DataType.FLOAT, 32, 16);
                    SDVariable weight = sd.constant("w", w);
                    SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32);
                    SDVariable h = sd.mmul("matmul", input, weight);
                    SDVariable out = sd.nn().relu("output", h, 0);
                    sd.setGraphExecutionMode(mode);

                    INDArray threadInput = Nd4j.randn(DataType.FLOAT, 1, 32);
                    Map<String, INDArray> ph = new LinkedHashMap<>();
                    ph.put("input", threadInput);

                    for (int step = 0; step < stepsPerThread; step++) {
                        threadInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
                        try {
                            Map<String, INDArray> result = sd.output(ph, "output");
                            INDArray output = result.get("output");
                            assertNotNull(output, mode + " thread=" + threadId + " step=" + step);
                            assertArrayEquals(new long[]{1, 16}, output.shape(),
                                    mode + " thread=" + threadId + " step=" + step + " shape");
                        } catch (Exception e) {
                            errors.incrementAndGet();
                            log.error("[MIXED_MODES {}] thread={} step={}: {}", mode, threadId, step, e.getMessage());
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    if (sd != null) sd.close();
                    doneLatch.countDown();
                }
            });
        }

        startLatch.countDown();
        assertTrue(doneLatch.await(60, TimeUnit.SECONDS), "Threads did not complete within 60s");
        executor.shutdown();

        assertEquals(0, errors.get(), "Mixed modes: " + errors.get() + " errors across " + numThreads + " threads");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 6: Shared graph with Triton-compiled ops under heavy concurrent load
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "concurrentTritonReplay_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON"})
    void testConcurrentTritonCompositeReplay(GraphExecutionMode mode) throws Exception {
        SameDiff sd = SameDiff.create();
        try {
            // Build graph with Triton-eligible ops (elementwise: max, divide)
            // and non-Triton ops (matmul = cuBLAS gap ops)
            // This creates a composite replay scenario: Triton islands + cuBLAS gaps
            INDArray w1 = Nd4j.randn(DataType.FLOAT, 32, 32);
            INDArray w2 = Nd4j.randn(DataType.FLOAT, 32, 16);
            SDVariable weight1 = sd.constant("w1", w1);
            SDVariable weight2 = sd.constant("w2", w2);
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32);

            SDVariable h = sd.mmul("h1", input, weight1);
            SDVariable norm = sd.math().norm2("norm", h, 1);
            SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5));
            SDVariable maxNorm = sd.math().max("maxNorm", norm, eps);
            SDVariable normalized = sd.math().div("normalized", h, maxNorm);
            SDVariable out = sd.mmul("output", normalized, weight2);

            sd.setGraphExecutionMode(mode);

            // Pre-warm to get through compilation + capture
            INDArray warmupInput = Nd4j.randn(DataType.FLOAT, 1, 32);
            Map<String, INDArray> warmupPh = new LinkedHashMap<>();
            warmupPh.put("input", warmupInput);
            for (int i = 0; i < 30; i++) {
                warmupInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
                sd.output(warmupPh, "output");
            }

            // Now concurrent replay
            int numThreads = 4;
            int stepsPerThread = 30;
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            AtomicInteger errors = new AtomicInteger(0);
            AtomicInteger nanErrors = new AtomicInteger(0);
            CountDownLatch startLatch = new CountDownLatch(1);
            CountDownLatch doneLatch = new CountDownLatch(numThreads);

            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    try {
                        startLatch.await();
                        INDArray threadInput = Nd4j.randn(DataType.FLOAT, 1, 32);
                        Map<String, INDArray> ph = new LinkedHashMap<>();
                        ph.put("input", threadInput);

                        for (int step = 0; step < stepsPerThread; step++) {
                            threadInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
                            try {
                                Map<String, INDArray> result = sd.output(ph, "output");
                                INDArray output = result.get("output");
                                if (output == null) {
                                    errors.incrementAndGet();
                                    log.error("[TRITON_REPLAY] thread={} step={}: null", threadId, step);
                                } else {
                                    assertArrayEquals(new long[]{1, 16}, output.shape(),
                                            "thread=" + threadId + " step=" + step + " shape");
                                    if (output.isNaN().any()) {
                                        nanErrors.incrementAndGet();
                                        log.error("[TRITON_REPLAY] thread={} step={}: NaN", threadId, step);
                                    }
                                }
                            } catch (Exception e) {
                                errors.incrementAndGet();
                                log.error("[TRITON_REPLAY] thread={} step={}: {}", threadId, step, e.getMessage());
                            }
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            startLatch.countDown();
            assertTrue(doneLatch.await(60, TimeUnit.SECONDS), "Threads did not complete within 60s");
            executor.shutdown();

            assertEquals(0, errors.get(), mode + ": " + errors.get() + " structural errors");
            assertEquals(0, nanErrors.get(), mode + ": " + nanErrors.get() + " NaN errors");
        } finally {
            sd.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 7: Output value correctness under concurrent execution
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "concurrentValueCorrectness_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testConcurrentValueCorrectness(GraphExecutionMode mode) throws Exception {
        // Build a deterministic graph: output = input * W
        // Verify each thread gets the correct output for its specific input
        SameDiff sd = SameDiff.create();
        try {
            INDArray w = Nd4j.ones(DataType.FLOAT, 4, 4).mul(2.0); // Simple weight: all 2.0
            SDVariable weight = sd.constant("w", w);
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 4);
            SDVariable out = sd.mmul("output", input, weight);

            sd.setGraphExecutionMode(mode);

            // Pre-warm
            INDArray warmupInput = Nd4j.ones(DataType.FLOAT, 1, 4);
            for (int i = 0; i < 10; i++) {
                sd.output(Collections.singletonMap("input", warmupInput), "output");
            }

            int numThreads = 4;
            int stepsPerThread = 15;
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            AtomicInteger valueErrors = new AtomicInteger(0);
            CountDownLatch startLatch = new CountDownLatch(1);
            CountDownLatch doneLatch = new CountDownLatch(numThreads);

            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    try {
                        startLatch.await();
                        // Each thread uses a distinct input value
                        float inputVal = (threadId + 1) * 1.0f;
                        INDArray threadInput = Nd4j.ones(DataType.FLOAT, 1, 4).mul(inputVal);
                        Map<String, INDArray> ph = new LinkedHashMap<>();
                        ph.put("input", threadInput);

                        // Expected: input * W = [val, val, val, val] * [[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2]]
                        // = [val*4*2, val*4*2, val*4*2, val*4*2] = [val*8, val*8, val*8, val*8]
                        float expected = inputVal * 4.0f * 2.0f;

                        for (int step = 0; step < stepsPerThread; step++) {
                            try {
                                Map<String, INDArray> result = sd.output(ph, "output");
                                INDArray output = result.get("output");
                                assertNotNull(output);

                                float[] values = output.dup().data().asFloat();
                                for (int i = 0; i < values.length; i++) {
                                    if (Math.abs(values[i] - expected) > 0.01f) {
                                        valueErrors.incrementAndGet();
                                        log.error("[VALUE] thread={} step={}: expected={} got={} (index={})",
                                                threadId, step, expected, values[i], i);
                                        break;
                                    }
                                }
                            } catch (Exception e) {
                                valueErrors.incrementAndGet();
                                log.error("[VALUE] thread={} step={}: {}", threadId, step, e.getMessage());
                            }
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            startLatch.countDown();
            assertTrue(doneLatch.await(60, TimeUnit.SECONDS), "Threads did not complete within 60s");
            executor.shutdown();

            assertEquals(0, valueErrors.get(),
                    mode + ": " + valueErrors.get() + " value correctness errors (concurrent cross-contamination)");
        } finally {
            sd.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 8: Concurrent markVariable while other threads are executing
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "concurrentMarkVariableDuringExecution_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testConcurrentMarkVariableDuringExecution(GraphExecutionMode mode) throws Exception {
        // This tests the critical race: one thread calls markVariable (triggering
        // invalidateSegmentCaptures which resets segment executionCount to 0 and
        // clears replay handles), while other threads are actively executing the
        // plan. The fix for this is anySegmentNeedsWarmup() which forces full
        // stream sync after invalidation instead of event-based sync.
        SameDiff sd = SameDiff.create();
        try {
            // Build graph: input -> matmul(weight) -> relu -> output
            INDArray w = Nd4j.randn(DataType.FLOAT, 32, 16);
            SDVariable weight = sd.var("weight", w);
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32);
            SDVariable h = sd.mmul("hidden", input, weight);
            SDVariable out = sd.nn().relu("output", h, 0);

            sd.setGraphExecutionMode(mode);

            // Pre-warm to get to REPLAYING
            INDArray warmupInput = Nd4j.randn(DataType.FLOAT, 1, 32);
            for (int i = 0; i < 10; i++) {
                warmupInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
                sd.output(Collections.singletonMap("input", warmupInput), "output");
            }

            // Launch executor threads that continuously call sd.output()
            int numExecutors = 3;
            int stepsPerExecutor = 30;
            ExecutorService executor = Executors.newFixedThreadPool(numExecutors + 1); // +1 for invalidator
            AtomicInteger execErrors = new AtomicInteger(0);
            AtomicInteger nanErrors = new AtomicInteger(0);
            AtomicInteger staleErrors = new AtomicInteger(0);
            CountDownLatch startLatch = new CountDownLatch(1);
            CountDownLatch doneLatch = new CountDownLatch(numExecutors + 1);

            // Executor threads: continuously run sd.output() with changing inputs
            for (int t = 0; t < numExecutors; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    try {
                        startLatch.await();
                        INDArray threadInput = Nd4j.randn(DataType.FLOAT, 1, 32);
                        Map<String, INDArray> ph = new LinkedHashMap<>();
                        ph.put("input", threadInput);
                        INDArray prevOutput = null;

                        for (int step = 0; step < stepsPerExecutor; step++) {
                            threadInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
                            try {
                                Map<String, INDArray> result = sd.output(ph, "output");
                                INDArray output = result.get("output");
                                if (output == null) {
                                    execErrors.incrementAndGet();
                                    log.error("[EXEC_T{}] step {}: null output", threadId, step);
                                    continue;
                                }
                                if (output.isNaN().any()) {
                                    nanErrors.incrementAndGet();
                                    log.error("[EXEC_T{}] step {}: NaN in output", threadId, step);
                                }
                                // Check for stale output (step 0 = step 1 bug)
                                INDArray dup = output.dup();
                                if (prevOutput != null && dup.equalsWithEps(prevOutput, 1e-6)) {
                                    staleErrors.incrementAndGet();
                                    log.error("[EXEC_T{}] step {}: stale output (identical to previous)",
                                            threadId, step);
                                }
                                prevOutput = dup;
                            } catch (Exception e) {
                                execErrors.incrementAndGet();
                                log.error("[EXEC_T{}] step {}: {}", threadId, step, e.getMessage());
                            }
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            // Invalidator thread: periodically calls markVariable on the weight
            // via DspHandle — this triggers invalidateSegmentCaptures on the native plan
            final DspHandle dspHandle = sd.dsp();
            final int weightExtIdx = dspHandle.extInputIndex("weight");
            executor.submit(() -> {
                try {
                    startLatch.await();
                    // Give executor threads a head start
                    Thread.sleep(10);
                    for (int cycle = 0; cycle < 5; cycle++) {
                        try {
                            // Mark weight as variable — triggers invalidateSegmentCaptures
                            dspHandle.markVariable(weightExtIdx);
                            Thread.sleep(20); // Let executor threads run during invalidation
                        } catch (Exception e) {
                            log.error("[INVALIDATOR] cycle {}: {}", cycle, e.getMessage());
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    doneLatch.countDown();
                }
            });

            startLatch.countDown();
            assertTrue(doneLatch.await(120, TimeUnit.SECONDS),
                    "Threads did not complete within 120s — possible deadlock");
            executor.shutdown();

            assertEquals(0, execErrors.get(),
                    mode + ": " + execErrors.get() + " execution errors during concurrent markVariable");
            assertEquals(0, nanErrors.get(),
                    mode + ": " + nanErrors.get() + " NaN errors during concurrent markVariable");
            // Stale errors are acceptable during the invalidation transition step
            // but should not persist. We allow up to numExecutors * numCycles stale
            // outputs (one per invalidation per thread).
            int maxAllowedStale = numExecutors * 5;
            assertTrue(staleErrors.get() <= maxAllowedStale,
                    mode + ": " + staleErrors.get() + " stale outputs (max allowed: " + maxAllowedStale + ")");
        } finally {
            sd.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 9: Concurrent re-capture after markVariable unseal
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "concurrentRecaptureAfterMarkVariable_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testConcurrentRecaptureAfterMarkVariable(GraphExecutionMode mode) throws Exception {
        // Tests the race where markVariable unseals the plan, multiple threads
        // simultaneously reach the warmup/re-capture threshold, and all try to
        // re-capture CUDA graphs. Only one should win; others must wait or skip.
        SameDiff sd = SameDiff.create();
        try {
            INDArray w = Nd4j.randn(DataType.FLOAT, 16, 8);
            SDVariable weight = sd.var("weight", w);
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 16);
            SDVariable out = sd.mmul("output", input, weight);

            sd.setGraphExecutionMode(mode);

            // Pre-warm to REPLAYING
            INDArray warmupInput = Nd4j.randn(DataType.FLOAT, 1, 16);
            for (int i = 0; i < 10; i++) {
                warmupInput.assign(Nd4j.randn(DataType.FLOAT, 1, 16));
                sd.output(Collections.singletonMap("input", warmupInput), "output");
            }

            // Now invalidate via DspHandle — triggers unseal
            DspHandle dspHandle = sd.dsp();
            dspHandle.markVariable(dspHandle.extInputIndex("weight"));

            // Launch threads that all start simultaneously after invalidation
            // They all race through warmup → capture → replay
            int numThreads = 4;
            int stepsPerThread = 20;
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            AtomicInteger errors = new AtomicInteger(0);
            CountDownLatch startLatch = new CountDownLatch(1);
            CountDownLatch doneLatch = new CountDownLatch(numThreads);

            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    try {
                        startLatch.await();
                        INDArray threadInput = Nd4j.randn(DataType.FLOAT, 1, 16);
                        Map<String, INDArray> ph = new LinkedHashMap<>();
                        ph.put("input", threadInput);
                        INDArray prevOutput = null;

                        for (int step = 0; step < stepsPerThread; step++) {
                            threadInput.assign(Nd4j.randn(DataType.FLOAT, 1, 16));
                            try {
                                Map<String, INDArray> result = sd.output(ph, "output");
                                INDArray output = result.get("output");
                                assertNotNull(output, "T" + threadId + " step " + step);
                                assertFalse(output.isNaN().any(),
                                        "T" + threadId + " step " + step + " NaN");

                                // Verify output changes with input (no stuck tokens)
                                INDArray dup = output.dup();
                                if (prevOutput != null && step > 2) {
                                    // After warmup, outputs must differ
                                    assertFalse(dup.equalsWithEps(prevOutput, 1e-6),
                                            "T" + threadId + " step " + step + " stale (same as step " + (step - 1) + ")");
                                }
                                prevOutput = dup;
                            } catch (Exception e) {
                                errors.incrementAndGet();
                                log.error("[RECAPTURE_T{}] step {}: {}", threadId, step, e.getMessage());
                            }
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            startLatch.countDown();
            assertTrue(doneLatch.await(120, TimeUnit.SECONDS),
                    "Re-capture threads did not complete within 120s — possible deadlock");
            executor.shutdown();

            assertEquals(0, errors.get(),
                    mode + ": " + errors.get() + " errors during concurrent re-capture after markVariable");
        } finally {
            sd.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 10: Concurrent changing inputs WITHOUT invalidator
    // Isolates: is staleness from changing inputs concurrently, or from markVariable?
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "concurrentChangingInputs_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testConcurrentChangingInputsNoInvalidator(GraphExecutionMode mode) throws Exception {
        SameDiff sd = SameDiff.create();
        try {
            INDArray w = Nd4j.randn(DataType.FLOAT, 32, 16);
            SDVariable weight = sd.var("weight", w);
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32);
            SDVariable h = sd.mmul("hidden", input, weight);
            SDVariable out = sd.nn().relu("output", h, 0);

            sd.setGraphExecutionMode(mode);

            // Pre-warm
            INDArray warmupInput = Nd4j.randn(DataType.FLOAT, 1, 32);
            for (int i = 0; i < 10; i++) {
                warmupInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
                sd.output(Collections.singletonMap("input", warmupInput), "output");
            }

            int numThreads = 3;
            int stepsPerThread = 30;
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            AtomicInteger staleErrors = new AtomicInteger(0);
            AtomicInteger execErrors = new AtomicInteger(0);
            CountDownLatch startLatch = new CountDownLatch(1);
            CountDownLatch doneLatch = new CountDownLatch(numThreads);

            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    try {
                        startLatch.await();
                        INDArray threadInput = Nd4j.randn(DataType.FLOAT, 1, 32);
                        Map<String, INDArray> ph = new LinkedHashMap<>();
                        ph.put("input", threadInput);
                        INDArray prevOutput = null;

                        for (int step = 0; step < stepsPerThread; step++) {
                            threadInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
                            try {
                                Map<String, INDArray> result = sd.output(ph, "output");
                                INDArray output = result.get("output");
                                if (output == null) {
                                    execErrors.incrementAndGet();
                                    continue;
                                }
                                INDArray dup = output.dup();
                                if (prevOutput != null && dup.equalsWithEps(prevOutput, 1e-6)) {
                                    staleErrors.incrementAndGet();
                                    log.error("[CHG_T{}] step {}: stale output", threadId, step);
                                }
                                prevOutput = dup;
                            } catch (Exception e) {
                                execErrors.incrementAndGet();
                                log.error("[CHG_T{}] step {}: {}", threadId, step, e.getMessage());
                            }
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            startLatch.countDown();
            assertTrue(doneLatch.await(60, TimeUnit.SECONDS),
                    "Threads did not complete within 60s");
            executor.shutdown();

            assertEquals(0, execErrors.get(),
                    mode + ": " + execErrors.get() + " execution errors");
            assertEquals(0, staleErrors.get(),
                    mode + ": " + staleErrors.get() + " stale outputs (changing inputs, no invalidator)");
        } finally {
            sd.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 11: markVariable with FIXED inputs (no assign per step)
    // Isolates: is staleness from markVariable invalidation, or input changes?
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "markVariableFixedInputs_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testMarkVariableWithFixedInputs(GraphExecutionMode mode) throws Exception {
        SameDiff sd = SameDiff.create();
        try {
            INDArray w = Nd4j.randn(DataType.FLOAT, 32, 16);
            SDVariable weight = sd.var("weight", w);
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32);
            SDVariable h = sd.mmul("hidden", input, weight);
            SDVariable out = sd.nn().relu("output", h, 0);

            sd.setGraphExecutionMode(mode);

            // Pre-warm
            INDArray warmupInput = Nd4j.randn(DataType.FLOAT, 1, 32);
            for (int i = 0; i < 10; i++) {
                warmupInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
                sd.output(Collections.singletonMap("input", warmupInput), "output");
            }

            int numExecutors = 3;
            int stepsPerExecutor = 30;
            ExecutorService executor = Executors.newFixedThreadPool(numExecutors + 1);
            AtomicInteger valueErrors = new AtomicInteger(0);
            AtomicInteger execErrors = new AtomicInteger(0);
            CountDownLatch startLatch = new CountDownLatch(1);
            CountDownLatch doneLatch = new CountDownLatch(numExecutors + 1);

            // Each executor thread uses a FIXED input — never changes
            for (int t = 0; t < numExecutors; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    try {
                        startLatch.await();
                        // Fixed input per thread — compute expected output
                        float val = (threadId + 1) * 3.0f;
                        INDArray threadInput = Nd4j.ones(DataType.FLOAT, 1, 32).mul(val);
                        Map<String, INDArray> ph = new LinkedHashMap<>();
                        ph.put("input", threadInput);

                        // Compute expected via direct matmul + relu
                        INDArray expected = org.nd4j.linalg.ops.transforms.Transforms.relu(
                                threadInput.mmul(w), false);

                        for (int step = 0; step < stepsPerExecutor; step++) {
                            try {
                                Map<String, INDArray> result = sd.output(ph, "output");
                                INDArray output = result.get("output");
                                if (output == null) {
                                    execErrors.incrementAndGet();
                                    continue;
                                }
                                INDArray dup = output.dup();
                                if (!dup.equalsWithEps(expected, 0.1)) {
                                    valueErrors.incrementAndGet();
                                    log.error("[MV_T{}] step {}: wrong value. expected first={} got first={}",
                                            threadId, step,
                                            expected.getFloat(0), dup.getFloat(0));
                                }
                            } catch (Exception e) {
                                execErrors.incrementAndGet();
                                log.error("[MV_T{}] step {}: {}", threadId, step, e.getMessage());
                            }
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            // Invalidator thread: calls markVariable periodically
            final DspHandle dspHandle = sd.dsp();
            final int weightExtIdx = dspHandle.extInputIndex("weight");
            executor.submit(() -> {
                try {
                    startLatch.await();
                    Thread.sleep(10);
                    for (int cycle = 0; cycle < 5; cycle++) {
                        try {
                            dspHandle.markVariable(weightExtIdx);
                            Thread.sleep(20);
                        } catch (Exception e) {
                            log.error("[INVALIDATOR] cycle {}: {}", cycle, e.getMessage());
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    doneLatch.countDown();
                }
            });

            startLatch.countDown();
            assertTrue(doneLatch.await(120, TimeUnit.SECONDS),
                    "Threads did not complete within 120s");
            executor.shutdown();

            assertEquals(0, execErrors.get(),
                    mode + ": " + execErrors.get() + " execution errors");
            // After markVariable, the weight hasn't actually changed value —
            // so outputs should STILL be correct
            assertEquals(0, valueErrors.get(),
                    mode + ": " + valueErrors.get() + " value errors (markVariable + fixed inputs)");
        } finally {
            sd.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 12: Verify output buffer identity isolation across threads
    // Each thread should get its OWN output array, never aliased to another thread
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "outputBufferIsolation_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testOutputBufferIsolation(GraphExecutionMode mode) throws Exception {
        SameDiff sd = SameDiff.create();
        try {
            INDArray w = Nd4j.eye(4).castTo(DataType.FLOAT); // identity matrix
            SDVariable weight = sd.constant("w", w);
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 4);
            SDVariable out = sd.mmul("output", input, weight);

            sd.setGraphExecutionMode(mode);

            // Pre-warm
            for (int i = 0; i < 10; i++) {
                sd.output(Collections.singletonMap("input", Nd4j.ones(DataType.FLOAT, 1, 4)), "output");
            }

            int numThreads = 4;
            int stepsPerThread = 20;
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            AtomicInteger aliasErrors = new AtomicInteger(0);
            AtomicInteger valueErrors = new AtomicInteger(0);
            CountDownLatch startLatch = new CountDownLatch(1);
            CountDownLatch doneLatch = new CountDownLatch(numThreads);

            // Collect all output buffer addresses to check for aliasing
            ConcurrentHashMap<Long, Integer> bufferAddresses = new ConcurrentHashMap<>();

            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    try {
                        startLatch.await();
                        // Identity matrix weight means output = input
                        float val = (threadId + 1) * 10.0f;
                        INDArray threadInput = Nd4j.ones(DataType.FLOAT, 1, 4).mul(val);
                        Map<String, INDArray> ph = new LinkedHashMap<>();
                        ph.put("input", threadInput);

                        for (int step = 0; step < stepsPerThread; step++) {
                            Map<String, INDArray> result = sd.output(ph, "output");
                            INDArray output = result.get("output");
                            INDArray dup = output.dup();

                            // Verify value: output should equal input (identity weight)
                            float got = dup.getFloat(0);
                            if (Math.abs(got - val) > 0.01f) {
                                valueErrors.incrementAndGet();
                                // Check if it matches another thread's value
                                int otherThread = -1;
                                for (int ot = 0; ot < numThreads; ot++) {
                                    if (ot != threadId) {
                                        float otherVal = (ot + 1) * 10.0f;
                                        if (Math.abs(got - otherVal) < 0.01f) {
                                            otherThread = ot;
                                            break;
                                        }
                                    }
                                }
                                if (otherThread >= 0) {
                                    log.error("[ISOLATION_T{}] step {}: got T{}'s value {}! (expected {})",
                                            threadId, step, otherThread, got, val);
                                    aliasErrors.incrementAndGet();
                                } else {
                                    log.error("[ISOLATION_T{}] step {}: wrong value {} (expected {})",
                                            threadId, step, got, val);
                                }
                            }

                            // Check buffer address for aliasing
                            long addr = output.data().address();
                            Integer owner = bufferAddresses.putIfAbsent(addr, threadId);
                            if (owner != null && owner != threadId) {
                                log.error("[ISOLATION_T{}] step {}: output buffer 0x{} belongs to T{}!",
                                        threadId, step, Long.toHexString(addr), owner);
                                aliasErrors.incrementAndGet();
                            }
                        }
                    } catch (Exception e) {
                        log.error("[ISOLATION_T{}]: {}", threadId, e.getMessage(), e);
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            startLatch.countDown();
            assertTrue(doneLatch.await(60, TimeUnit.SECONDS),
                    "Threads did not complete within 60s");
            executor.shutdown();

            assertEquals(0, aliasErrors.get(),
                    mode + ": " + aliasErrors.get() + " buffer aliasing errors (cross-thread contamination)");
            assertEquals(0, valueErrors.get(),
                    mode + ": " + valueErrors.get() + " value errors");
        } finally {
            sd.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 13: Single pool thread with changing inputs — NO main-thread warmup
    // Isolates: is the issue from cross-thread session, or from pool thread itself?
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "singlePoolThreadChangingInputs_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testSinglePoolThreadChangingInputsNoMainWarmup(GraphExecutionMode mode) throws Exception {
        SameDiff sd = SameDiff.create();
        try {
            INDArray w = Nd4j.randn(DataType.FLOAT, 32, 16);
            SDVariable weight = sd.var("weight", w);
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32);
            SDVariable h = sd.mmul("hidden", input, weight);
            SDVariable out = sd.nn().relu("output", h, 0);

            sd.setGraphExecutionMode(mode);

            // NO main-thread warmup — pool thread does everything
            ExecutorService executor = Executors.newSingleThreadExecutor();
            AtomicInteger staleErrors = new AtomicInteger(0);
            AtomicInteger execErrors = new AtomicInteger(0);

            Future<?> future = executor.submit(() -> {
                INDArray threadInput = Nd4j.randn(DataType.FLOAT, 1, 32);
                Map<String, INDArray> ph = new LinkedHashMap<>();
                ph.put("input", threadInput);
                INDArray prevOutput = null;

                for (int step = 0; step < 30; step++) {
                    threadInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
                    try {
                        Map<String, INDArray> result = sd.output(ph, "output");
                        INDArray output = result.get("output");
                        if (output == null) {
                            execErrors.incrementAndGet();
                            continue;
                        }
                        INDArray dup = output.dup();
                        if (prevOutput != null && dup.equalsWithEps(prevOutput, 1e-6)) {
                            staleErrors.incrementAndGet();
                            log.error("[POOL_SINGLE] step {}: stale output", step);
                        }
                        prevOutput = dup;
                    } catch (Exception e) {
                        execErrors.incrementAndGet();
                        log.error("[POOL_SINGLE] step {}: {}", step, e.getMessage());
                    }
                }
            });

            future.get(60, TimeUnit.SECONDS);
            executor.shutdown();

            log.info("singlePoolThread result: {} stale, {} errors", staleErrors.get(), execErrors.get());
            assertEquals(0, execErrors.get(),
                    mode + ": " + execErrors.get() + " execution errors");
            assertEquals(0, staleErrors.get(),
                    mode + ": " + staleErrors.get() + " stale outputs (single pool thread, no main warmup)");
        } finally {
            sd.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 14: Main thread only — changing inputs, EXACT same graph as pool tests
    // Control: if this passes but pool thread fails, the issue is pool-thread-specific
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "mainThreadChangingInputs_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testMainThreadChangingInputs(GraphExecutionMode mode) throws Exception {
        SameDiff sd = SameDiff.create();
        try {
            INDArray w = Nd4j.randn(DataType.FLOAT, 32, 16);
            SDVariable weight = sd.var("weight", w);
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32);
            SDVariable h = sd.mmul("hidden", input, weight);
            SDVariable out = sd.nn().relu("output", h, 0);

            sd.setGraphExecutionMode(mode);

            INDArray threadInput = Nd4j.randn(DataType.FLOAT, 1, 32);
            Map<String, INDArray> ph = new LinkedHashMap<>();
            ph.put("input", threadInput);
            INDArray prevOutput = null;
            int staleErrors = 0;

            for (int step = 0; step < 30; step++) {
                threadInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
                Map<String, INDArray> result = sd.output(ph, "output");
                INDArray output = result.get("output");
                assertNotNull(output, "step " + step + ": null output");
                INDArray dup = output.dup();
                if (prevOutput != null && dup.equalsWithEps(prevOutput, 1e-6)) {
                    staleErrors++;
                    log.error("[MAIN] step {}: stale output", step);
                }
                prevOutput = dup;
            }

            assertEquals(0, staleErrors,
                    mode + ": " + staleErrors + " stale outputs on main thread");
        } finally {
            sd.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 15: Pool thread — per-step output validation against reference
    // Instead of just detecting stale, compute expected output manually
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "poolThreadOutputValidation_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testPoolThreadOutputValidation(GraphExecutionMode mode) throws Exception {
        SameDiff sd = SameDiff.create();
        try {
            INDArray w = Nd4j.randn(DataType.FLOAT, 32, 16);
            SDVariable weight = sd.var("weight", w);
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32);
            SDVariable h = sd.mmul("hidden", input, weight);
            SDVariable out = sd.nn().relu("output", h, 0);

            sd.setGraphExecutionMode(mode);

            ExecutorService executor = Executors.newSingleThreadExecutor();
            AtomicInteger wrongOutputs = new AtomicInteger(0);
            AtomicInteger staleOutputs = new AtomicInteger(0);
            AtomicInteger firstWrongStep = new AtomicInteger(-1);

            Future<?> future = executor.submit(() -> {
                INDArray threadInput = Nd4j.randn(DataType.FLOAT, 1, 32);
                Map<String, INDArray> ph = new LinkedHashMap<>();
                ph.put("input", threadInput);
                INDArray prevOutput = null;

                for (int step = 0; step < 30; step++) {
                    threadInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));

                    // Compute expected: relu(input @ weight)
                    INDArray expected = threadInput.mmul(w);
                    // relu in-place
                    for (long j = 0; j < expected.length(); j++) {
                        float v = expected.getFloat(j);
                        if (v < 0) expected.putScalar(j, 0.0f);
                    }

                    Map<String, INDArray> result = sd.output(ph, "output");
                    INDArray actual = result.get("output");
                    if (actual == null) {
                        wrongOutputs.incrementAndGet();
                        continue;
                    }

                    INDArray dup = actual.dup();

                    // Check staleness
                    if (prevOutput != null && dup.equalsWithEps(prevOutput, 1e-6)) {
                        staleOutputs.incrementAndGet();
                    }

                    // Check correctness against manual computation
                    double maxDiff = 0;
                    for (long j = 0; j < dup.length(); j++) {
                        maxDiff = Math.max(maxDiff,
                                Math.abs(dup.getFloat(j) - expected.getFloat(j)));
                    }
                    if (maxDiff > 0.01) {
                        wrongOutputs.incrementAndGet();
                        if (firstWrongStep.get() < 0) firstWrongStep.set(step);
                        log.error("[POOL_VALIDATE] step {}: maxDiff={} (WRONG OUTPUT)", step, maxDiff);
                    }

                    prevOutput = dup;
                }
            });

            future.get(60, TimeUnit.SECONDS);
            executor.shutdown();

            log.info("poolThreadOutputValidation {}: {} wrong, {} stale, firstWrong={}",
                    mode, wrongOutputs.get(), staleOutputs.get(), firstWrongStep.get());
            assertEquals(0, wrongOutputs.get(),
                    mode + ": " + wrongOutputs.get() + " wrong outputs (first at step " + firstWrongStep.get() + ")");
            assertEquals(0, staleOutputs.get(),
                    mode + ": " + staleOutputs.get() + " stale outputs");
        } finally {
            sd.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 16: Pool thread — main thread warmup first, THEN pool thread executes
    // Isolates: does main-thread warmup fix the pool thread issue?
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "mainWarmupPoolExecute_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testMainWarmupThenPoolThreadExecute(GraphExecutionMode mode) throws Exception {
        SameDiff sd = SameDiff.create();
        try {
            INDArray w = Nd4j.randn(DataType.FLOAT, 32, 16);
            SDVariable weight = sd.var("weight", w);
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32);
            SDVariable h = sd.mmul("hidden", input, weight);
            SDVariable out = sd.nn().relu("output", h, 0);

            sd.setGraphExecutionMode(mode);

            // Main thread warmup — go through all DSP phases
            INDArray warmupInput = Nd4j.randn(DataType.FLOAT, 1, 32);
            Map<String, INDArray> warmupPh = new LinkedHashMap<>();
            warmupPh.put("input", warmupInput);
            for (int i = 0; i < 10; i++) {
                warmupInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
                sd.output(warmupPh, "output");
            }

            // Now pool thread runs with changing inputs
            ExecutorService executor = Executors.newSingleThreadExecutor();
            AtomicInteger staleErrors = new AtomicInteger(0);

            Future<?> future = executor.submit(() -> {
                INDArray threadInput = Nd4j.randn(DataType.FLOAT, 1, 32);
                Map<String, INDArray> ph = new LinkedHashMap<>();
                ph.put("input", threadInput);
                INDArray prevOutput = null;

                for (int step = 0; step < 30; step++) {
                    threadInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
                    Map<String, INDArray> result = sd.output(ph, "output");
                    INDArray output = result.get("output");
                    if (output == null) continue;
                    INDArray dup = output.dup();
                    if (prevOutput != null && dup.equalsWithEps(prevOutput, 1e-6)) {
                        staleErrors.incrementAndGet();
                        log.error("[MAIN_WARMUP_POOL] step {}: stale output", step);
                    }
                    prevOutput = dup;
                }
            });

            future.get(60, TimeUnit.SECONDS);
            executor.shutdown();

            assertEquals(0, staleErrors.get(),
                    mode + ": " + staleErrors.get() + " stale outputs (main warmup, pool execute)");
        } finally {
            sd.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 17: Pool thread with GPU-heavy graph (NOT captureProducedNoKernels)
    // Uses add op with large tensors to guarantee GPU kernels during capture.
    // Isolates: is the bug specific to captureProducedNoKernels path?
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "poolThreadGpuHeavyGraph_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testPoolThreadGpuHeavyGraph(GraphExecutionMode mode) throws Exception {
        SameDiff sd = SameDiff.create();
        try {
            // Use large tensors and element-wise ops that WILL produce GPU kernels
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 1024);
            SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 1024, 512));
            SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 512, 256));
            SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 256));
            SDVariable h1 = sd.mmul("h1", input, w1);
            SDVariable h2 = sd.math().abs("abs_h1", h1);  // element-wise = GPU kernel
            SDVariable h3 = sd.mmul("h2", h2, w2);
            SDVariable out = sd.math().add("output", h3, bias);  // element-wise = GPU kernel

            sd.setGraphExecutionMode(mode);

            ExecutorService executor = Executors.newSingleThreadExecutor();
            AtomicInteger staleErrors = new AtomicInteger(0);

            Future<?> future = executor.submit(() -> {
                INDArray threadInput = Nd4j.randn(DataType.FLOAT, 1, 1024);
                Map<String, INDArray> ph = new LinkedHashMap<>();
                ph.put("input", threadInput);
                INDArray prevOutput = null;

                for (int step = 0; step < 30; step++) {
                    threadInput.assign(Nd4j.randn(DataType.FLOAT, 1, 1024));
                    Nd4j.getExecutioner().commit();
                    Map<String, INDArray> result = sd.output(ph, "output");
                    INDArray output = result.get("output");
                    if (output == null) continue;
                    INDArray dup = output.dup();
                    // Diagnostic: print first 4 output values for each step
                    float[] first4 = new float[Math.min(4, (int)dup.length())];
                    for (int f = 0; f < first4.length; f++) first4[f] = dup.getFloat(f);
                    log.info("[POOL_GPU_HEAVY] step {} first4={}", step, java.util.Arrays.toString(first4));
                    if (prevOutput != null && dup.equalsWithEps(prevOutput, 1e-6)) {
                        staleErrors.incrementAndGet();
                        float[] prev4 = new float[Math.min(4, (int)prevOutput.length())];
                        for (int f = 0; f < prev4.length; f++) prev4[f] = prevOutput.getFloat(f);
                        log.error("[POOL_GPU_HEAVY] step {}: STALE output cur={} prev={}", step,
                                java.util.Arrays.toString(first4), java.util.Arrays.toString(prev4));
                    }
                    prevOutput = dup;
                }
            });

            future.get(60, TimeUnit.SECONDS);
            executor.shutdown();

            assertEquals(0, staleErrors.get(),
                    mode + ": " + staleErrors.get() + " stale outputs (pool thread, GPU-heavy graph)");
        } finally {
            sd.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 18: Pool thread — first output correct, then track WHICH step goes stale
    // Records at exactly which execution count staleness begins
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "poolThreadStalenessTiming_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    void testPoolThreadStalenessTiming(GraphExecutionMode mode) throws Exception {
        SameDiff sd = SameDiff.create();
        try {
            INDArray w = Nd4j.randn(DataType.FLOAT, 32, 16);
            SDVariable weight = sd.var("weight", w);
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32);
            SDVariable h = sd.mmul("hidden", input, weight);
            SDVariable out = sd.nn().relu("output", h, 0);

            sd.setGraphExecutionMode(mode);

            ExecutorService executor = Executors.newSingleThreadExecutor();
            List<String> stepLog = Collections.synchronizedList(new ArrayList<>());
            AtomicInteger firstStaleStep = new AtomicInteger(-1);
            AtomicInteger totalStale = new AtomicInteger(0);

            Future<?> future = executor.submit(() -> {
                INDArray threadInput = Nd4j.randn(DataType.FLOAT, 1, 32);
                Map<String, INDArray> ph = new LinkedHashMap<>();
                ph.put("input", threadInput);
                INDArray prevOutput = null;

                for (int step = 0; step < 30; step++) {
                    threadInput.assign(Nd4j.randn(DataType.FLOAT, 1, 32));
                    Map<String, INDArray> result = sd.output(ph, "output");
                    INDArray output = result.get("output");
                    if (output == null) {
                        stepLog.add("step=" + step + " NULL");
                        continue;
                    }
                    INDArray dup = output.dup();
                    float sum = dup.sumNumber().floatValue();
                    boolean stale = prevOutput != null && dup.equalsWithEps(prevOutput, 1e-6);
                    stepLog.add("step=" + step + " sum=" + String.format("%.4f", sum)
                            + (stale ? " STALE" : " OK"));
                    if (stale) {
                        totalStale.incrementAndGet();
                        firstStaleStep.compareAndSet(-1, step);
                    }
                    prevOutput = dup;
                }
            });

            future.get(60, TimeUnit.SECONDS);
            executor.shutdown();

            // Print ALL steps for diagnosis
            log.info("=== {} staleness timing ===", mode);
            for (String s : stepLog) {
                log.info("  {}", s);
            }
            log.info("firstStale={} totalStale={}", firstStaleStep.get(), totalStale.get());

            assertEquals(0, totalStale.get(),
                    mode + ": " + totalStale.get() + " stale outputs (first at step " + firstStaleStep.get() + ")");
        } finally {
            sd.close();
        }
    }
}
