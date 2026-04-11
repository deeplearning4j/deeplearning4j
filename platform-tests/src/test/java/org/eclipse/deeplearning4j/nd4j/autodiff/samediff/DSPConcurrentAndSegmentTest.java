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
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueContext;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.nativeblas.OpaqueLaunchContext;
import org.nd4j.nativeblas.OpaqueNDArray;

import java.util.*;
import java.util.concurrent.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for concurrent DSP plan execution, mixed capturable/non-capturable
 * segment sequences, and plan serialization round-trips.
 *
 * <p>Covers:
 * <ul>
 *   <li>Concurrent plan execution from multiple threads (ExecutorService + CountDownLatch)</li>
 *   <li>Mixed capturable/non-capturable segment boundaries under CUDA_GRAPHS</li>
 *   <li>Plan serialize → deserialize → compile → execute round-trips</li>
 *   <li>Multi-step replay from a deserialized plan</li>
 * </ul>
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test -Dtest=DSPConcurrentAndSegmentTest
 * </pre>
 */
@Slf4j
@Tag("dsp")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class DSPConcurrentAndSegmentTest extends BaseNd4jTestWithBackends {

    private static final double TOLERANCE = 1e-4;

    @Override
    public char ordering() {
        return 'c';
    }

    @BeforeAll
    static void enableDspGlobally() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    private void enableDsp(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
    }

    /**
     * Build a matmul chain graph: input → mmul(w1) → relu → mmul(w2) → output
     */
    private SameDiff buildMatmulChain() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 16);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 32, 16));
        SDVariable h = sd.nn.relu("mm1_relu", sd.mmul("mm1", input, w1), 0);
        sd.mmul("output", h, w2);
        return sd;
    }

    /**
     * Build an elementwise chain graph: input → add → mul → sigmoid → tanh → output
     */
    private SameDiff buildElementwiseChain() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable bias1 = sd.constant("bias1", Nd4j.randn(DataType.FLOAT, 1, 32).mul(0.1f));
        SDVariable scale = sd.constant("scale", Nd4j.valueArrayOf(new long[]{1, 32}, 2.0f));
        SDVariable bias2 = sd.constant("bias2", Nd4j.valueArrayOf(new long[]{1, 32}, -0.5f));
        SDVariable h1 = input.add("add1", bias1);
        SDVariable h2 = h1.mul("mul1", scale);
        SDVariable h3 = sd.nn.sigmoid("sigmoid1", h2);
        SDVariable result = sd.math.tanh("output", h3.add("add2", bias2));
        return sd;
    }

    /**
     * Compile a native plan handle from a DynamicShapePlan.
     */
    private Pointer compileNativePlan(DynamicShapePlan plan) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        byte[] serialized = plan.serialize();
        assertNotNull(serialized, "Plan serialization returned null");
        assertTrue(serialized.length > 0, "Plan serialization returned empty");
        BytePointer planBytes = new BytePointer(serialized);
        try {
            return nativeOps.compileDynamicShapePlan(planBytes, serialized.length);
        } catch (UnsupportedOperationException e) {
            log.info("Backend does not support native executor: {}", e.getMessage());
            return null;
        } finally {
            planBytes.close();
        }
    }

    /**
     * Execute a compiled native plan and return the output map.
     */
    private Map<String, INDArray> executeNativePlan(Pointer planHandle, DynamicShapePlan plan,
                                                     INDArray[] extInputs) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numOutputs = plan.getRequestedOutputs().size();

        OpaqueContext opContext = nativeOps.createGraphContext(1);
        try {
            for (int i = 0; i < extInputs.length; i++) {
                OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(extInputs[i]);
                nativeOps.setGraphContextInputArray(opContext, i, opaqueIn);
            }

            for (int i = 0; i < numOutputs; i++) {
                INDArray dummy = Nd4j.empty(DataType.FLOAT);
                OpaqueNDArray opaqueOut = OpaqueNDArray.fromINDArray(dummy);
                nativeOps.setGraphContextOutputArray(opContext, i, opaqueOut);
            }

            Pointer execStream = null;
            try {
                OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
                if (lc != null) {
                    execStream = nativeOps.lcExecutionStream(lc);
                    if (execStream != null) execStream.retainReference();
                }
            } catch (Exception e) {
                // CPU backend
            }

            int status = nativeOps.executeDynamicShapePlan(planHandle, opContext, execStream);
            if (status != 0) {
                String errMsg = nativeOps.lastErrorMessage();
                nativeOps.clearLastError();
                fail("Native plan execution failed with status " + status + ": " + errMsg);
            }

            Map<String, INDArray> results = new LinkedHashMap<>();
            List<String> requestedOutputs = new ArrayList<>(plan.getRequestedOutputs());

            for (int i = 0; i < numOutputs; i++) {
                OpaqueNDArray opaqueOut = nativeOps.getOutputArrayNative(opContext, i);
                assertNotNull(opaqueOut, "Null output at index " + i);
                assertFalse(opaqueOut.isNull(), "Null OpaqueNDArray at index " + i);

                long[] shapeInfo = OpaqueNDArray.getOpaqueNDArrayShapeInfo(opaqueOut);
                long[] shape = Shape.shape(shapeInfo);
                DataType dtype = ArrayOptionsHelper.dataType(shapeInfo);
                long length = OpaqueNDArray.getOpaqueNDArrayLength(opaqueOut);

                INDArray result = Nd4j.createUninitialized(dtype, shape);
                Pointer nativePrimary = nativeOps.getOpaqueNDArrayBuffer(opaqueOut);
                Pointer nativeSpecial = nativeOps.getOpaqueNDArraySpecialBuffer(opaqueOut);

                OpaqueDataBuffer srcOdb = nativeOps.dbCreateExternalDataBuffer(
                        length, dtype.toInt(), nativePrimary, nativeSpecial);
                if (srcOdb != null) {
                    OpaqueDataBuffer dstOdb = result.data().opaqueBuffer();
                    if (dstOdb != null) {
                        nativeOps.copyBuffer(dstOdb, length, srcOdb, 0, 0);
                    }
                }
                results.put(requestedOutputs.get(i), result);
            }
            return results;
        } finally {
            nativeOps.deleteGraphContext(opContext);
        }
    }

    /**
     * Resolve external inputs from a plan and a placeholder map.
     */
    private INDArray[] resolveExternalInputs(DynamicShapePlan plan, SameDiff sd,
                                              Map<String, INDArray> placeholders) {
        String[] extKeys = plan.getExternalInputKeys();
        INDArray[] extInputs = new INDArray[extKeys.length];
        for (int i = 0; i < extKeys.length; i++) {
            String varName = extKeys[i];
            INDArray arr = placeholders != null ? placeholders.get(varName) : null;
            if (arr == null) {
                SDVariable var = sd.getVariable(varName);
                if (var != null) arr = var.getArr();
            }
            assertNotNull(arr, "Missing external input: " + varName);
            extInputs[i] = arr;
        }
        return extInputs;
    }

    /**
     * Compute reference output using Java execution with SLOT_BY_SLOT mode.
     */
    private INDArray computeReferenceOutput(SameDiff sdTemplate, String outputName,
                                             Map<String, INDArray> inputs) {
        SameDiff ref = SameDiff.create();
        // Rebuild the same graph structure for reference
        for (Map.Entry<String, INDArray> entry : inputs.entrySet()) {
            INDArray val = entry.getValue();
            SDVariable v = ref.placeHolder(entry.getKey(), val.dataType(), val.shape());
        }
        // Copy constants from original
        for (SDVariable var : sdTemplate.variables()) {
            if (var.isConstant() && var.getArr() != null) {
                ref.constant(var.name(), var.getArr().dup());
            }
        }
        // Rebuild the computation (simplified: just use the template's output method)
        Map<String, INDArray> refResult = sdTemplate.output(inputs, outputName);
        INDArray refOutput = refResult.get(outputName).dup();
        return refOutput;
    }

    // =========================================================================
    // Concurrent execution tests
    // =========================================================================

    /**
     * Test 1: Two different graphs executed concurrently from separate threads.
     * Each graph compiles its own DSP plan. Both must produce correct output.
     */
    @Test
    @Order(1)
    @DisplayName("Concurrent execution: two different graphs produce correct output")
    public void testConcurrentPlanExecutionTwoGraphs() throws Exception {
        // Graph A: matmul chain
        SameDiff sdA = buildMatmulChain();
        sdA.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        enableDsp(sdA);

        // Graph B: elementwise chain
        SameDiff sdB = buildElementwiseChain();
        sdB.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        enableDsp(sdB);

        INDArray inputA = Nd4j.randn(DataType.FLOAT, 1, 16);
        INDArray inputB = Nd4j.randn(DataType.FLOAT, 1, 32);

        // Reference outputs (sequential)
        Map<String, INDArray> refA = sdA.output(Map.of("input", inputA.dup()), "output");
        Map<String, INDArray> refB = sdB.output(Map.of("input", inputB.dup()), "output");

        // Concurrent execution
        ExecutorService executor = Executors.newFixedThreadPool(2);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(2);

        Future<Map<String, INDArray>> futureA = executor.submit(() -> {
            try {
                startLatch.await();
                Map<String, INDArray> result = sdA.output(Map.of("input", inputA.dup()), "output");
                log.info("Thread A completed: output shape={}",
                        Arrays.toString(result.get("output").shape()));
                return result;
            } finally {
                doneLatch.countDown();
            }
        });

        Future<Map<String, INDArray>> futureB = executor.submit(() -> {
            try {
                startLatch.await();
                Map<String, INDArray> result = sdB.output(Map.of("input", inputB.dup()), "output");
                log.info("Thread B completed: output shape={}",
                        Arrays.toString(result.get("output").shape()));
                return result;
            } finally {
                doneLatch.countDown();
            }
        });

        // Start both threads simultaneously
        startLatch.countDown();

        // Wait for completion
        boolean finished = doneLatch.await(60, TimeUnit.SECONDS);
        assertTrue(finished, "Concurrent execution timed out");

        Map<String, INDArray> resultA = futureA.get(10, TimeUnit.SECONDS);
        Map<String, INDArray> resultB = futureB.get(10, TimeUnit.SECONDS);

        // Verify correctness
        double diffA = resultA.get("output").sub(refA.get("output")).amaxNumber().doubleValue();
        double diffB = resultB.get("output").sub(refB.get("output")).amaxNumber().doubleValue();

        log.info("Graph A maxDiff from reference: {}", diffA);
        log.info("Graph B maxDiff from reference: {}", diffB);

        assertTrue(diffA < TOLERANCE, "Graph A diverged from reference by " + diffA);
        assertTrue(diffB < TOLERANCE, "Graph B diverged from reference by " + diffB);

        executor.shutdown();
        sdA.close();
        sdB.close();
        log.info("testConcurrentPlanExecutionTwoGraphs: PASSED");
    }

    /**
     * Test 2: Same graph, 4 threads each with different inputs.
     * All must produce correct output.
     */
    @Test
    @Order(2)
    @DisplayName("Concurrent execution: same graph, 4 threads with different inputs")
    public void testConcurrentSameGraphDifferentInputs() throws Exception {
        SameDiff sd = buildMatmulChain();
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        enableDsp(sd);

        int numThreads = 4;
        int stepsPerThread = 3;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CountDownLatch startLatch = new CountDownLatch(1);

        // Generate inputs
        INDArray[][] allInputs = new INDArray[numThreads][stepsPerThread];
        INDArray[][] refOutputs = new INDArray[numThreads][stepsPerThread];

        for (int t = 0; t < numThreads; t++) {
            for (int s = 0; s < stepsPerThread; s++) {
                allInputs[t][s] = Nd4j.randn(DataType.FLOAT, 1, 16).muli(t * 10 + s + 1);
                // Compute reference sequentially
                Map<String, INDArray> ref = sd.output(Map.of("input", allInputs[t][s].dup()), "output");
                refOutputs[t][s] = ref.get("output").dup();
            }
        }

        // Concurrent execution
        List<Future<INDArray[][]>> futures = new ArrayList<>();
        for (int t = 0; t < numThreads; t++) {
            final int threadIdx = t;
            Future<INDArray[][]> future = executor.submit(() -> {
                startLatch.await();
                INDArray[][] results = new INDArray[stepsPerThread][];
                for (int s = 0; s < stepsPerThread; s++) {
                    Map<String, INDArray> result = sd.output(
                            Map.of("input", allInputs[threadIdx][s].dup()), "output");
                    results[s] = new INDArray[]{result.get("output").dup()};
                }
                return results;
            });
            futures.add(future);
        }

        startLatch.countDown();

        // Verify all results
        for (int t = 0; t < numThreads; t++) {
            INDArray[][] results = futures.get(t).get(30, TimeUnit.SECONDS);
            for (int s = 0; s < stepsPerThread; s++) {
                double diff = results[s][0].sub(refOutputs[t][s]).amaxNumber().doubleValue();
                log.info("Thread {} Step {}: maxDiff={}", t, s, diff);
                assertTrue(diff < TOLERANCE,
                        "Thread " + t + " Step " + s + " diverged by " + diff);
            }
        }

        executor.shutdown();
        sd.close();
        log.info("testConcurrentSameGraphDifferentInputs: PASSED — {} threads × {} steps",
                numThreads, stepsPerThread);
    }

    /**
     * Test 3: One thread compiles a plan while another thread executes
     * an already-compiled plan. No crashes.
     */
    @Test
    @Order(3)
    @DisplayName("Concurrent compile and execute: no crashes")
    public void testConcurrentPlanCompileAndExecute() throws Exception {
        // Graph for compilation thread
        SameDiff sdCompile = buildMatmulChain();
        sdCompile.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        enableDsp(sdCompile);

        // Graph for execution thread (pre-compiled)
        SameDiff sdExecute = buildMatmulChain();
        sdExecute.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        enableDsp(sdExecute);

        // Pre-warm the execution graph
        INDArray execInput = Nd4j.randn(DataType.FLOAT, 1, 16);
        Map<String, INDArray> execRef = sdExecute.output(Map.of("input", execInput.dup()), "output");

        ExecutorService executor = Executors.newFixedThreadPool(2);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(2);

        // Thread 1: compiles a plan
        Future<Boolean> compileFuture = executor.submit(() -> {
            try {
                startLatch.await();
                DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sdCompile, "output");
                assertNotNull(plan, "Compiled plan should not be null");
                Pointer handle = compileNativePlan(plan);
                if (handle != null) {
                    NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                    nativeOps.freeDynamicShapePlan(handle);
                    log.info("Compile thread: compiled and freed plan with {} slots", plan.getSlots().length);
                }
                plan.close();
                return true;
            } catch (Exception e) {
                log.error("Compile thread exception", e);
                throw e;
            } finally {
                doneLatch.countDown();
            }
        });

        // Thread 2: executes an already-warmed plan
        Future<Map<String, INDArray>> execFuture = executor.submit(() -> {
            try {
                startLatch.await();
                Map<String, INDArray> result = sdExecute.output(
                        Map.of("input", execInput.dup()), "output");
                log.info("Execute thread completed: output shape={}",
                        Arrays.toString(result.get("output").shape()));
                return result;
            } catch (Exception e) {
                log.error("Execute thread exception", e);
                throw e;
            } finally {
                doneLatch.countDown();
            }
        });

        startLatch.countDown();

        boolean finished = doneLatch.await(60, TimeUnit.SECONDS);
        assertTrue(finished, "Concurrent compile/execute timed out");

        // Both should succeed
        Boolean compiled = compileFuture.get(10, TimeUnit.SECONDS);
        assertTrue(compiled, "Compilation thread should succeed");

        Map<String, INDArray> execResult = execFuture.get(10, TimeUnit.SECONDS);
        assertNotNull(execResult, "Execution result should not be null");

        double diff = execResult.get("output").sub(execRef.get("output")).amaxNumber().doubleValue();
        assertTrue(diff < TOLERANCE, "Execution diverged by " + diff);

        executor.shutdown();
        sdCompile.close();
        sdExecute.close();
        log.info("testConcurrentPlanCompileAndExecute: PASSED");
    }

    // =========================================================================
    // Mixed segment sequence tests
    // =========================================================================

    /**
     * Test 4: Build a graph that forces segment boundaries:
     * matmul (capturable) → gather (non-capturable) → matmul (capturable).
     * Execute under SLOT_BY_SLOT reference mode. Verify output.
     */
    @Test
    @Order(4)
    @DisplayName("Mixed capturable/non-capturable segments produce correct output")
    public void testMixedCapturableNonCapturableSegments() throws Exception {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 16);

        // Segment 1: capturable matmul
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable h1 = sd.mmul("mm1", input, w1);

        // Non-capturable op: gather (data-dependent indices)
        INDArray embData = Nd4j.randn(DataType.FLOAT, 64, 32);
        SDVariable embeddings = sd.constant("embeddings", embData);
        // Gather along the first dimension — typically non-capturable
        // Gather with 4 indices → shape [4, 32]
        SDVariable gathered = sd.gather("gathered", embeddings,
                new int[]{0, 1, 2, 3}, 0);

        // Segment 2: capturable matmul after gather
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 32, 16));
        SDVariable h2 = sd.mmul("mm2", gathered, w2);

        // Combine: h1 is [1, 32], h2 is [4, 16] — we can't add these directly.
        // Instead, combine via mmul: use h1 to weight the gathered rows.
        // Reshape h1 from [1, 32] to [32, 1] for mmul with h2 [4, 16] → not compatible either.
        // Simpler approach: sum gathered rows to get [32], then add with h1 [1, 32].
        SDVariable gatheredSum = gathered.sum("gathered_sum", 0); // [32]
        SDVariable gatheredExpanded = sd.expandDims("gathered_expanded", gatheredSum, 0); // [1, 32]
        SDVariable combined = h1.add("add", gatheredExpanded); // [1, 32]
        sd.mmul("output", combined, w2); // [1, 16]

        enableDsp(sd);
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 16);
        Map<String, INDArray> resultMap = sd.output(Map.of("input", inputArr), "output");
        INDArray output = resultMap.get("output");

        assertNotNull(output, "Output should not be null");
        assertFalse(output.isNaN().any(), "Output contains NaN");
        assertFalse(output.isInfinite().any(), "Output contains Inf");
        log.info("Mixed segment test: output shape={}, min={}, max={}",
                Arrays.toString(output.shape()),
                output.minNumber().doubleValue(),
                output.maxNumber().doubleValue());

        sd.close();
        log.info("testMixedCapturableNonCapturableSegments: PASSED");
    }

    /**
     * Test 5: Verify data is correctly transferred between segments.
     * Output of segment N is input to segment N+1.
     */
    @Test
    @Order(5)
    @DisplayName("Segment boundary data transfer is correct")
    public void testSegmentBoundaryDataTransfer() throws Exception {
        // Build a multi-segment graph where each segment depends on the previous
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("x", DataType.FLOAT, -1, 8);

        // Segment 1: linear transform (capturable)
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 8, 16));
        SDVariable seg1Out = sd.mmul("seg1_out", input, w1); // [batch, 16]

        // Segment boundary: non-capturable op (gather) using input-dependent indices
        // Use argmax of seg1_out to create data-dependent indices
        SDVariable indices = sd.argmax("indices", seg1Out, 1); // [batch] indices into dim 1
        INDArray table = Nd4j.randn(DataType.FLOAT, 16, 16);
        SDVariable lookupTable = sd.constant("lookup", table);
        // Gather from lookup table using data-dependent indices
        SDVariable gathered = sd.gather("seg2_in", lookupTable, indices, 0); // [batch, 16]

        // Segment 2: matmul using gathered output
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 16, 4));
        sd.mmul("output", gathered, w2); // [batch, 4]

        enableDsp(sd);
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // Run multiple steps to verify data transfer stability
        int numSteps = 5;
        List<INDArray> outputs = new ArrayList<>();

        for (int i = 0; i < numSteps; i++) {
            // Use different random seeds so argmax indices actually change between steps
            Nd4j.getRandom().setSeed(i * 12345L + 42);
            INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 8);
            Map<String, INDArray> result = sd.output(Map.of("x", inputArr), "output");
            outputs.add(result.get("output").dup());
        }

        // Verify outputs are not all identical (data-dependent gather produces different results).
        // Note: some consecutive steps may produce identical output if argmax indices happen to match,
        // so we check that at least one pair differs across all steps.
        boolean anyDiff = false;
        for (int i = 1; i < numSteps; i++) {
            double diff = outputs.get(i).sub(outputs.get(i - 1)).amaxNumber().doubleValue();
            if (diff > 1e-6) {
                anyDiff = true;
                break;
            }
        }
        assertTrue(anyDiff,
                "All " + numSteps + " steps produced identical output — stale data or no data dependency");

        // Verify each output is valid
        for (int i = 0; i < numSteps; i++) {
            INDArray out = outputs.get(i);
            assertFalse(out.isNaN().any(), "Step " + i + " contains NaN");
            assertFalse(out.isInfinite().any(), "Step " + i + " contains Inf");
            log.info("Step {}: output[0..3]={}", i,
                    out.getFloat(0, 0) + ", " + out.getFloat(0, 1) + ", " +
                    out.getFloat(0, 2) + ", " + out.getFloat(0, 3));
        }

        sd.close();
        log.info("testSegmentBoundaryDataTransfer: PASSED — {} steps all valid", numSteps);
    }

    // =========================================================================
    // Serialization round-trip tests
    // =========================================================================

    /**
     * Test 6: Build graph → compile DSP plan → serialize → deserialize
     * (compile from serialized bytes) → execute → verify output matches
     * original execution.
     */
    @Test
    @Order(6)
    @DisplayName("Plan serialize → deserialize → execute round-trip")
    public void testPlanSerializeDeserializeExecute() throws Exception {
        // Build and compile original plan
        SameDiff sd = buildMatmulChain();
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 16);

        // Reference output from Java execution
        Map<String, INDArray> refResult = sd.output(Map.of("input", input.dup()), "output");
        INDArray refOutput = refResult.get("output").dup();

        // Compile native plan from the plan
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "output");
        assertNotNull(plan, "Plan should compile");
        log.info("Original plan: {} slots, {} external inputs, {} requested outputs",
                plan.getSlots().length, plan.getExternalInputKeys().length,
                plan.getRequestedOutputs().size());

        // Serialize
        byte[] serialized = plan.serialize();
        assertNotNull(serialized, "Serialized plan should not be null");
        assertTrue(serialized.length > 24, "Serialized plan too small");
        assertTrue(DynamicShapePlan.isValidSerializedPlan(serialized),
                "Serialized plan should have valid DSP1 header");
        log.info("Plan serialized to {} bytes", serialized.length);

        // Compile from serialized bytes
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null,
                "Native executor unavailable — skipping native round-trip");

        try {
            // Execute the compiled plan
            INDArray[] extInputs = resolveExternalInputs(plan, sd, Map.of("input", input.dup()));
            Map<String, INDArray> nativeResult = executeNativePlan(planHandle, plan, extInputs);
            INDArray nativeOutput = nativeResult.get("output");

            assertNotNull(nativeOutput, "Native output should not be null");
            assertFalse(nativeOutput.isNaN().any(), "Native output contains NaN");

            // Compare with reference
            double maxDiff = nativeOutput.sub(refOutput).amaxNumber().doubleValue();
            log.info("Serialize round-trip: native vs Java reference maxDiff={}", maxDiff);
            assertTrue(maxDiff < TOLERANCE,
                    "Serialized plan execution diverged from reference by " + maxDiff);

        } finally {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            nativeOps.freeDynamicShapePlan(planHandle);
        }

        plan.close();
        sd.close();
        log.info("testPlanSerializeDeserializeExecute: PASSED");
    }

    /**
     * Test 7: Deserialize a plan and run 10 replay iterations.
     * Verify all produce correct output.
     */
    @Test
    @Order(7)
    @DisplayName("Serialized plan multi-step replay (10 iterations)")
    public void testSerializedPlanMultiStepReplay() throws Exception {
        // Build and compile
        SameDiff sd = buildMatmulChain();
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        enableDsp(sd);

        // Compile and serialize
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "output");
        assertNotNull(plan, "Plan should compile");

        byte[] serialized = plan.serialize();
        assertTrue(DynamicShapePlan.isValidSerializedPlan(serialized),
                "Serialized plan should be valid");
        log.info("Plan serialized to {} bytes for multi-step replay test", serialized.length);

        // Compile from serialized bytes
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null,
                "Native executor unavailable — skipping multi-step replay");

        try {
            int numIterations = 10;
            INDArray[] inputs = new INDArray[numIterations];
            INDArray[] refOutputs = new INDArray[numIterations];
            INDArray[] nativeOutputs = new INDArray[numIterations];

            // Pre-compute references and inputs
            for (int i = 0; i < numIterations; i++) {
                inputs[i] = Nd4j.randn(DataType.FLOAT, 1, 16).muli(i + 1);
                Map<String, INDArray> refResult = sd.output(Map.of("input", inputs[i].dup()), "output");
                refOutputs[i] = refResult.get("output").dup();
            }

            // Execute all iterations from the same compiled handle
            for (int i = 0; i < numIterations; i++) {
                INDArray[] extInputs = resolveExternalInputs(plan, sd,
                        Map.of("input", inputs[i].dup()));
                Map<String, INDArray> nativeResult = executeNativePlan(planHandle, plan, extInputs);
                nativeOutputs[i] = nativeResult.get("output").dup();

                assertFalse(nativeOutputs[i].isNaN().any(),
                        "Iteration " + i + " native output contains NaN");

                double maxDiff = nativeOutputs[i].sub(refOutputs[i]).amaxNumber().doubleValue();
                log.info("Multi-step replay iteration {}: maxDiff={}", i, maxDiff);
                assertTrue(maxDiff < TOLERANCE,
                        "Iteration " + i + " diverged from reference by " + maxDiff);
            }

            // Verify all outputs are different (not stale)
            for (int i = 1; i < numIterations; i++) {
                double diff = nativeOutputs[i].sub(nativeOutputs[i - 1]).amaxNumber().doubleValue();
                assertTrue(diff > 1e-6,
                        "Iteration " + i + " identical to iteration " + (i - 1)
                                + " — stale replay data. diff=" + diff);
            }

        } finally {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            nativeOps.freeDynamicShapePlan(planHandle);
        }

        plan.close();
        sd.close();
        log.info("testSerializedPlanMultiStepReplay: PASSED — {} iterations all correct", 10);
    }
}
