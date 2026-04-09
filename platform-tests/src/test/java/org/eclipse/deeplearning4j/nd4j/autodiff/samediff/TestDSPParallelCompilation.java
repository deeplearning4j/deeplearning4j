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
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for thread-safety of parallel Triton compilation.
 *
 * <p>These tests create graphs with many segments that trigger the parallel
 * work-stealing compile pool in TritonGraphBackend_compile.cu. The goal is to
 * reproduce thread-safety races in:
 * <ul>
 *   <li>MLIR context creation (global dialect registry)</li>
 *   <li>CRC32 table initialization (check-then-act race)</li>
 *   <li>Static counters (irDumpCount, gridSyncCounter)</li>
 *   <li>OpCategoryTable static initialization across .cu TUs</li>
 * </ul>
 *
 * <p>The crash symptom is "double free or corruption (!prev)" / SIGABRT during
 * compilation when multiple threads race on MLIR internals.</p>
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test -Dtest=TestDSPParallelCompilation \
 *     -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestDSPParallelCompilation extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @BeforeAll
    static void enableDspGlobally() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    private void enableDsp(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
    }

    /**
     * Test 1: Wide graph — many independent linear chains create multiple segments.
     *
     * Graph structure: N independent chains, each with matmul→add→relu→matmul.
     * The non-capturable gather at the start of each chain creates segment
     * boundaries. With N=8 chains, we get 8+ segments compiled in parallel.
     *
     * This exercises the work-stealing pool with concurrent compileToGpuBinary
     * calls that each create an MLIR context + load dialects.
     */
    @Test
    @Order(1)
    @DisplayName("Parallel compilation: wide graph with 8 independent chains")
    public void testWideGraphParallelCompilation() {
        int numChains = 8;
        int dim = 16;
        int vocabSize = 32;

        SameDiff sd = SameDiff.create();

        // Shared embedding table
        INDArray embData = Nd4j.randn(DataType.FLOAT, vocabSize, dim);
        SDVariable embeddings = sd.constant("embeddings", embData);

        List<String> outputNames = new ArrayList<>();
        Map<String, INDArray> placeholders = new HashMap<>();

        for (int c = 0; c < numChains; c++) {
            String prefix = "chain" + c + "_";

            // Per-chain input indices (dynamic → non-capturable → forces segment boundary)
            SDVariable indices = sd.placeHolder(prefix + "idx", DataType.INT32, -1);
            SDVariable gathered = sd.gather(prefix + "gather", embeddings, indices, 0);

            // Capturable ops after gather
            INDArray wData = Nd4j.randn(DataType.FLOAT, dim, dim);
            INDArray bData = Nd4j.zeros(DataType.FLOAT, 1, dim);
            INDArray w2Data = Nd4j.randn(DataType.FLOAT, dim, dim);
            SDVariable w = sd.constant(prefix + "w1", wData);
            SDVariable b = sd.constant(prefix + "b1", bData);
            SDVariable w2 = sd.constant(prefix + "w2", w2Data);

            SDVariable h = sd.nn.relu(prefix + "relu",
                    sd.mmul(prefix + "mm1", gathered, w).add(prefix + "add1", b), 0);
            sd.mmul(prefix + "output", h, w2);

            outputNames.add(prefix + "output");
            placeholders.put(prefix + "idx", Nd4j.createFromArray(new int[]{c, c + 1, c + 2, c + 3}));
        }

        enableDsp(sd);

        // Execute — this triggers parallel compilation of all segments
        log.info("Executing wide graph with {} chains (expect parallel compilation)", numChains);
        Map<String, INDArray> results = sd.output(placeholders, outputNames);

        // Verify all outputs are valid (no NaN, no stale data)
        for (String name : outputNames) {
            INDArray output = results.get(name);
            assertNotNull(output, name + ": output is null");
            assertFalse(output.isNaN().any(), name + ": contains NaN");
            assertFalse(output.isInfinite().any(), name + ": contains Inf");
            float[] data = output.dup().data().asFloat();
            for (int i = 0; i < data.length; i++) {
                assertFalse(data[i] < -1e18f,
                        name + ": stale data at index " + i + " value=" + data[i]);
            }
            log.info("{}: shape={} min={} max={}", name,
                    Arrays.toString(output.shape()),
                    output.minNumber().floatValue(),
                    output.maxNumber().floatValue());
        }

        sd.close();
        log.info("testWideGraphParallelCompilation: PASSED — no crash, no corruption");
    }

    /**
     * Test 2: Deep chain — many sequential elementwise ops create a large segment
     * that may be split into sub-segments by the work-stealing pool.
     *
     * Graph: input → (relu → add_const → relu → add_const) × 20 → matmul → output
     * The single large segment gets sub-divided and compiled in parallel.
     */
    @Test
    @Order(2)
    @DisplayName("Parallel compilation: deep chain with 40+ ops in one segment")
    public void testDeepChainSubSegmentSplit() {
        int dim = 16;
        int depth = 20;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, dim);

        SDVariable current = x;
        for (int i = 0; i < depth; i++) {
            String prefix = "layer" + i + "_";
            INDArray biasData = Nd4j.rand(DataType.FLOAT, 1, dim).muli(0.01f);
            SDVariable bias = sd.constant(prefix + "bias", biasData);
            current = sd.nn.relu(prefix + "relu", current.add(prefix + "add", bias), 0);
        }

        INDArray wOut = Nd4j.randn(DataType.FLOAT, dim, 4);
        SDVariable w = sd.constant("wout", wOut);
        sd.mmul("output", current, w);

        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, dim);
        log.info("Executing deep chain with {} layers (expect sub-segment splitting)", depth);
        Map<String, INDArray> results = sd.output(Map.of("input", input), "output");

        INDArray output = results.get("output");
        assertNotNull(output, "output is null");
        assertFalse(output.isNaN().any(), "output contains NaN");
        assertFalse(output.isInfinite().any(), "output contains Inf");
        log.info("Output: shape={} min={} max={}",
                Arrays.toString(output.shape()),
                output.minNumber().floatValue(),
                output.maxNumber().floatValue());

        sd.close();
        log.info("testDeepChainSubSegmentSplit: PASSED — no crash, no corruption");
    }

    /**
     * Test 3: Repeated compilation — create and compile multiple independent graphs
     * sequentially. Each compilation creates new MLIR contexts. Tests that
     * context cleanup doesn't corrupt global MLIR state.
     */
    @Test
    @Order(3)
    @DisplayName("Parallel compilation: repeated graph creation + compilation")
    public void testRepeatedGraphCompilation() {
        int dim = 16;
        int numGraphs = 5;

        for (int g = 0; g < numGraphs; g++) {
            SameDiff sd = SameDiff.create();

            SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, dim);
            INDArray wData = Nd4j.randn(DataType.FLOAT, dim, dim);
            INDArray bData = Nd4j.zeros(DataType.FLOAT, 1, dim);
            SDVariable w = sd.constant("w", wData);
            SDVariable b = sd.constant("b", bData);
            SDVariable h = sd.nn.relu("relu", sd.mmul("mm", x, w).add("add", b), 0);

            // Add a second matmul to make it worth compiling
            INDArray w2Data = Nd4j.randn(DataType.FLOAT, dim, 4);
            SDVariable w2 = sd.constant("w2", w2Data);
            sd.mmul("output", h, w2);

            enableDsp(sd);

            INDArray input = Nd4j.randn(DataType.FLOAT, 2, dim);
            log.info("Graph {} of {}: executing", g + 1, numGraphs);
            Map<String, INDArray> results = sd.output(Map.of("input", input), "output");

            INDArray output = results.get("output");
            assertNotNull(output, "graph " + g + ": output is null");
            assertFalse(output.isNaN().any(), "graph " + g + ": contains NaN");

            // Verify against reference
            INDArray ref = Nd4j.nn.relu(input.mmul(wData).addi(bData), 0).mmul(w2Data);
            double maxDiff = output.sub(ref).amaxNumber().doubleValue();
            log.info("Graph {}: maxDiff from reference = {}", g + 1, maxDiff);
            assertTrue(maxDiff < 0.1, "graph " + g + ": diverged from reference by " + maxDiff);

            sd.close();
        }

        log.info("testRepeatedGraphCompilation: PASSED — {} graphs compiled cleanly", numGraphs);
    }

    /**
     * Test 4: Wide graph with frozen shapes — combines parallel compilation
     * stress with frozen-path execution. After compilation, freeze shapes
     * and execute multiple times.
     */
    @Test
    @Order(4)
    @DisplayName("Parallel compilation: wide graph + frozen shapes + multi-call")
    public void testWideGraphFrozenMultiCall() {
        int numChains = 6;
        int dim = 16;
        int vocabSize = 32;

        SameDiff sd = SameDiff.create();
        INDArray embData = Nd4j.randn(DataType.FLOAT, vocabSize, dim);
        SDVariable embeddings = sd.constant("embeddings", embData);

        List<String> outputNames = new ArrayList<>();
        Map<String, INDArray> placeholders = new HashMap<>();
        Map<String, INDArray> references = new HashMap<>();

        for (int c = 0; c < numChains; c++) {
            String prefix = "chain" + c + "_";
            SDVariable indices = sd.placeHolder(prefix + "idx", DataType.INT32, -1);
            SDVariable gathered = sd.gather(prefix + "gather", embeddings, indices, 0);

            INDArray wData = Nd4j.randn(DataType.FLOAT, dim, dim);
            INDArray bData = Nd4j.zeros(DataType.FLOAT, 1, dim);
            SDVariable w = sd.constant(prefix + "w", wData);
            SDVariable b = sd.constant(prefix + "b", bData);
            sd.mmul(prefix + "output", gathered, w).add(prefix + "out_bias", b);

            String outName = prefix + "out_bias";
            outputNames.add(outName);

            int[] idxArr = {c, c + 1, c + 2, c + 3};
            placeholders.put(prefix + "idx", Nd4j.createFromArray(idxArr));

            // Compute reference
            INDArray gatheredRef = Nd4j.create(DataType.FLOAT, 4, dim);
            for (int i = 0; i < idxArr.length; i++) {
                gatheredRef.putRow(i, embData.getRow(idxArr[i]));
            }
            references.put(outName, gatheredRef.mmul(wData).addi(bData));
        }

        enableDsp(sd);

        // Warmup (triggers compilation)
        log.info("Warmup execution (triggers parallel compilation)");
        Map<String, INDArray> results = sd.output(placeholders, outputNames);
        for (String name : outputNames) {
            double maxDiff = results.get(name).sub(references.get(name)).amaxNumber().doubleValue();
            log.info("Warmup {}: maxDiff={}", name, maxDiff);
            assertTrue(maxDiff < 0.1, name + ": warmup diverged by " + maxDiff);
        }

        // Freeze and execute 5 more times
        try {
            var sessionsField = SameDiff.class.getDeclaredField("sessions");
            sessionsField.setAccessible(true);
            @SuppressWarnings("unchecked")
            Map<Long, InferenceSession> sessions = (Map<Long, InferenceSession>) sessionsField.get(sd);
            InferenceSession session = sessions.get(Thread.currentThread().getId());
            if (session != null && session.getDynamicShapePlanExecutor() != null) {
                session.getDynamicShapePlanExecutor().setShapesFrozen(true);
                log.info("Froze DSP shapes");
            }
        } catch (Exception e) {
            log.warn("Could not freeze DSP shapes: {}", e.getMessage());
        }

        for (int call = 0; call < 5; call++) {
            results = sd.output(placeholders, outputNames);
            for (String name : outputNames) {
                INDArray output = results.get(name);
                assertFalse(output.isNaN().any(), "frozen call " + call + " " + name + ": NaN");
                double maxDiff = output.sub(references.get(name)).amaxNumber().doubleValue();
                log.info("Frozen call {} {}: maxDiff={}", call + 1, name, maxDiff);
                assertTrue(maxDiff < 0.1,
                        "frozen call " + call + " " + name + ": diverged by " + maxDiff);
            }
        }

        sd.close();
        log.info("testWideGraphFrozenMultiCall: PASSED — compilation + frozen execution clean");
    }

    /**
     * Test 5: Stress test — create many small independent graphs in rapid
     * succession, each compiled from scratch. This maximizes contention
     * on MLIR global state.
     */
    @Test
    @Order(5)
    @DisplayName("Parallel compilation: rapid-fire 10 small graphs")
    public void testRapidFireCompilation() {
        int dim = 8;
        int numGraphs = 10;

        for (int g = 0; g < numGraphs; g++) {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, dim);

            // Random depth 2-5 for variety
            int depth = 2 + (g % 4);
            SDVariable current = x;
            for (int d = 0; d < depth; d++) {
                INDArray wData = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
                SDVariable w = sd.constant("w" + d, wData);
                current = sd.nn.relu("relu" + d, sd.mmul("mm" + d, current, w), 0);
            }
            current.rename("output");

            enableDsp(sd);

            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
            Map<String, INDArray> result = sd.output(Map.of("x", input), "output");

            INDArray output = result.get("output");
            assertNotNull(output, "graph " + g + ": null output");
            assertFalse(output.isNaN().any(), "graph " + g + ": NaN");

            sd.close();
        }

        log.info("testRapidFireCompilation: PASSED — {} graphs compiled without crash", numGraphs);
    }

    /**
     * Test 6: Gap ops after CUDA graph replay.
     *
     * Graph: input → gather(non-capturable) → permute(gap) → matmul → output
     *
     * The gather creates a segment boundary. The capturable segment contains
     * matmul, but permute is a gap op (not compiled by Triton). After CUDA
     * graph capture, replay only runs the Triton kernel — permute must be
     * re-executed after replay for downstream ops to see valid data.
     *
     * This is the exact pattern that caused the vision encoder SIGABRT.
     */
    @Test
    @Order(6)
    @DisplayName("Gap ops re-executed after CUDA graph replay")
    public void testGapOpsAfterReplay() {
        int dim = 16;
        int vocabSize = 32;

        SameDiff sd = SameDiff.create();
        INDArray embData = Nd4j.randn(DataType.FLOAT, vocabSize, dim);
        SDVariable embeddings = sd.constant("embeddings", embData);
        SDVariable indices = sd.placeHolder("indices", DataType.INT32, -1);

        // gather → reshape → permute (gap op) → matmul
        SDVariable gathered = sd.gather("gathered", embeddings, indices, 0);
        // Reshape to [2, 2, dim] to enable permute
        SDVariable reshaped = sd.reshape("reshaped", gathered, 2, 2, dim);
        // Permute [2, 2, dim] → [2, 2, dim] (swap first two dims)
        SDVariable permuted = sd.permute("permuted", reshaped, 1, 0, 2);

        // Downstream matmul that reads from permuted output
        INDArray wData = Nd4j.randn(DataType.FLOAT, dim, 4);
        SDVariable w = sd.constant("w", wData);
        // Reshape to [4, dim] for matmul
        SDVariable flat = sd.reshape("flat", permuted, 4, dim);
        sd.mmul("output", flat, w);

        enableDsp(sd);

        INDArray idx = Nd4j.createFromArray(new int[]{0, 5, 10, 15});

        // Compute reference
        INDArray gatheredRef = Nd4j.create(DataType.FLOAT, 4, dim);
        int[] idxArr = {0, 5, 10, 15};
        for (int i = 0; i < idxArr.length; i++) {
            gatheredRef.putRow(i, embData.getRow(idxArr[i]));
        }
        // reshape [4,dim] → [2,2,dim], permute [2,2,dim] → [2,2,dim], flatten → [4,dim]
        INDArray reshapedRef = gatheredRef.reshape(2, 2, dim);
        INDArray permutedRef = reshapedRef.permute(1, 0, 2).dup();
        INDArray flatRef = permutedRef.reshape(4, dim);
        INDArray ref = flatRef.mmul(wData);

        // Warmup (unfrozen)
        Map<String, INDArray> result0 = sd.output(Map.of("indices", idx), "output");
        INDArray output0 = result0.get("output");
        assertNotNull(output0, "warmup output is null");
        assertFalse(output0.isNaN().any(), "warmup: NaN");
        double diff0 = output0.sub(ref).amaxNumber().doubleValue();
        log.info("Warmup: maxDiff={}", diff0);
        assertTrue(diff0 < 0.1, "warmup diverged: " + diff0);

        // Freeze shapes
        try {
            var sessionsField = SameDiff.class.getDeclaredField("sessions");
            sessionsField.setAccessible(true);
            @SuppressWarnings("unchecked")
            Map<Long, InferenceSession> sessions = (Map<Long, InferenceSession>) sessionsField.get(sd);
            InferenceSession session = sessions.get(Thread.currentThread().getId());
            if (session != null && session.getDynamicShapePlanExecutor() != null) {
                session.getDynamicShapePlanExecutor().setShapesFrozen(true);
                log.info("Froze DSP shapes");
            }
        } catch (Exception e) {
            log.warn("Could not freeze: {}", e.getMessage());
        }

        // Execute 5 times frozen — mixed gap work must remain ordered and correct
        // across the frozen replay schedule.
        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> result = sd.output(Map.of("indices", idx), "output");
            INDArray output = result.get("output");
            assertNotNull(output, "frozen call " + (i + 1) + ": null output");
            assertFalse(output.isNaN().any(), "frozen call " + (i + 1) + ": NaN");
            float[] data = output.dup().data().asFloat();
            for (int j = 0; j < data.length; j++) {
                assertFalse(data[j] < -1e18f,
                        "frozen call " + (i + 1) + ": stale sentinel at index " + j + " = " + data[j]);
            }
            double diff = output.sub(ref).amaxNumber().doubleValue();
            log.info("Frozen call {}: maxDiff={}", i + 1, diff);
            assertTrue(diff < 0.1, "frozen call " + (i + 1) + " diverged: " + diff);
        }

        sd.close();
        log.info("testGapOpsAfterReplay: PASSED — mixed gap work remained correct during frozen execution");
    }
}
