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
package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.DecoderUtils;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;
import java.util.concurrent.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * CUDA Graph Replay Regression Tests.
 *
 * These tests reproduce the VLM decode loop that hangs on step 2 (first CUDA graph replay).
 * The decode loop is:
 *   Step 0: prefill (regular execution)
 *   Step 1: first decode (DSP captures CUDA graph)
 *   Step 2+: replay (CUDA graph replay — THIS IS WHERE THE HANG OCCURS)
 *
 * Each test runs with a 90-second timeout so it fails fast instead of hanging forever.
 *
 * The test structure exercises the EXACT code path used by kompile's VLM server:
 * - DSP auto-compile enabled (native)
 * - outputDirect from step 2+
 * - Static KV cache with scatter updates
 * - Present KV output closing when not frozen
 */
@Slf4j
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@DisplayName("CUDA Graph Replay Regression")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestCudaGraphReplayRegression {

    private SameDiff decoder;
    private SameDiff embedTokens;
    private INDArray embeddingTable;
    private String logitsName;
    private List<String> kvOutputNames;
    private List<String> allOutputNames;
    private List<String> keyNames;
    private List<String> valueNames;

    private static final int MAX_STEPS = 30;  // enough to exercise replay path well
    private static final int TIMEOUT_SECONDS = 90;

    @BeforeAll
    public void loadModel() throws Exception {
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);

        decoder = OnnxModelCache.importWithCache(decoderResult.getModelFile().getAbsolutePath());
        embedTokens = OnnxModelCache.importWithCache(embedResult.getModelFile().getAbsolutePath());

        // Find embedding table (largest 2D constant)
        embeddingTable = null;
        long bestRows = 0;
        for (SDVariable var : embedTokens.variables()) {
            INDArray arr = embedTokens.getArrForVarName(var.name());
            if (arr != null && arr.rank() == 2 && arr.size(0) > bestRows) {
                bestRows = arr.size(0);
                embeddingTable = arr;
            }
        }
        assertNotNull(embeddingTable, "Could not find embedding table");

        // Find output names
        kvOutputNames = new ArrayList<>();
        for (String name : decoder.outputs()) {
            if (name.contains("logit")) {
                logitsName = name;
            } else if (name.startsWith("present")) {
                kvOutputNames.add(name);
            }
        }
        if (logitsName == null) logitsName = decoder.outputs().get(0);

        allOutputNames = new ArrayList<>();
        allOutputNames.add(logitsName);
        allOutputNames.addAll(kvOutputNames);

        keyNames = new ArrayList<>();
        valueNames = new ArrayList<>();
        for (String kvName : kvOutputNames) {
            if (kvName.contains(".key")) keyNames.add(kvName);
            else if (kvName.contains(".value")) valueNames.add(kvName);
        }

        log.info("Loaded: inputs={} outputs={} kvPairs={}",
                decoder.inputs().size(), decoder.outputs().size(), keyNames.size());
    }

    @AfterAll
    public void cleanup() {
        if (decoder != null) decoder.close();
        if (embedTokens != null) embedTokens.close();
    }

    private INDArray buildPrefillEmbeddings(int[] tokens) {
        long hiddenSize = embeddingTable.size(1);
        INDArray result = Nd4j.create(DataType.FLOAT, 1, tokens.length, hiddenSize);
        for (int i = 0; i < tokens.length; i++) {
            result.get(NDArrayIndex.point(0), NDArrayIndex.point(i), NDArrayIndex.all())
                    .assign(embeddingTable.getRow(tokens[i]));
        }
        return result;
    }

    /**
     * Run the full decode loop (prefill + N decode steps) with DSP + CUDA graph replay.
     * Returns the list of generated token IDs, or throws TimeoutException if hung.
     */
    private List<Integer> runDecodeLoop(int maxSteps) throws Exception {
        long hiddenSize = embeddingTable.size(1);
        int[] promptTokens = {0, 44, 2692, 46};  // <bos> This is a
        INDArray prefillEmbeds = buildPrefillEmbeddings(promptTokens);
        long seqLen = prefillEmbeds.shape()[1];

        decoder.resetSession();
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        ArrayCacheMemoryMgr.setGrowthFactor(1.0);

        // Step 0: Prefill
        Map<String, INDArray> inputs = DecoderUtils.buildDecoderInputMap(
                decoder.inputs(), decoder,
                prefillEmbeds, null,
                0, seqLen, null, 0, 0,
                false, hiddenSize);

        Map<String, INDArray> result = decoder.output(inputs, allOutputNames.toArray(new String[0]));
        int firstToken = result.get(logitsName).get(NDArrayIndex.point(0),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all()).argMax().getInt(0);

        List<Integer> generated = new ArrayList<>();
        generated.add(firstToken);
        log.info("Step 0 (prefill): token={}", firstToken);

        long maxKvLen = seqLen + maxSteps;
        Map<String, INDArray> staticKvBuffers = DecoderUtils.padKvCacheToStaticSize(
                result, keyNames, valueNames, maxKvLen);
        Nd4j.getExecutioner().commit();
        long cachePos = seqLen;
        long pastSeqLen = seqLen;

        // Clear prefill DSP plan
        {
            InferenceSession sess = decoder.getOrCreateSession();
            DynamicShapePlanExecutor exec = sess.getDynamicShapePlanExecutor();
            if (exec != null && exec.getCurrentPlan() != null) {
                decoder.clearDynamicShapePlanCache();
                sess.clearAllCaches();
            }
        }

        // Decode loop: step 1 = capture, step 2+ = replay
        for (int step = 1; step < maxSteps; step++) {
            int prevToken = generated.get(generated.size() - 1);
            INDArray tokenEmbed = embeddingTable.getRow(prevToken).reshape(1, 1, hiddenSize);
            INDArray inputIds = Nd4j.createFromArray(new int[]{prevToken}).reshape(1, 1).castTo(DataType.LONG);

            Map<String, INDArray> decInputs = DecoderUtils.buildDecoderInputMap(
                    decoder.inputs(), decoder,
                    tokenEmbed, inputIds,
                    pastSeqLen, 1,
                    staticKvBuffers, maxKvLen, cachePos,
                    true, hiddenSize, null, false);

            long t0 = System.currentTimeMillis();

            // outputDirect from step 2 (like StaticKvCacheDecodeLoop)
            boolean useDirect = step >= 2;
            if (useDirect) {
                result = decoder.outputDirect(decInputs, allOutputNames.toArray(new String[0]));
            } else {
                result = decoder.output(decInputs, allOutputNames.toArray(new String[0]));
            }

            long elapsed = System.currentTimeMillis() - t0;

            int nextToken = result.get(logitsName).get(NDArrayIndex.point(0),
                    NDArrayIndex.point(0), NDArrayIndex.all()).argMax().getInt(0);
            generated.add(nextToken);

            // Check frozen state
            boolean nativePlanFrozen = false;
            {
                InferenceSession sess = decoder.getOrCreateSession();
                DynamicShapePlanExecutor exec = sess.getDynamicShapePlanExecutor();
                nativePlanFrozen = exec != null && exec.isShapesFrozen();
            }

            String phaseLabel = step == 1 ? "capture" : (nativePlanFrozen ? "replay(frozen)" : "replay");
            log.info("Step {} ({}): token={} elapsed={}ms", step, phaseLabel, nextToken, elapsed);

            // Scatter KV entries
            DecoderUtils.scatterNewKvEntries(staticKvBuffers, result, keyNames, valueNames, maxKvLen, cachePos);
            cachePos++;

            // Close present KV outputs when not frozen (like StaticKvCacheDecodeLoop)
            if (!nativePlanFrozen) {
                for (String pn : keyNames) {
                    INDArray pv = result.get(pn);
                    if (pv != null && !pv.wasClosed() && pv.data() != null) {
                        pv.setCloseable(true);
                        pv.close();
                    }
                }
                for (String pn : valueNames) {
                    INDArray pv = result.get(pn);
                    if (pv != null && !pv.wasClosed() && pv.data() != null) {
                        pv.setCloseable(true);
                        pv.close();
                    }
                }
            }

            pastSeqLen++;
        }

        return generated;
    }

    // ========================================================================
    // Test 1: Full decode loop with all changes — the main regression test
    // ========================================================================

    /**
     * Tests the complete decode loop path that matches kompile VLM server behavior.
     * Step 2 is the FIRST CUDA graph replay — this is where the hang occurs.
     * Uses a timeout to fail fast instead of hanging.
     */
    @Test
    @Order(1)
    @Timeout(value = TIMEOUT_SECONDS, unit = TimeUnit.SECONDS)
    @DisplayName("1. Full decode loop — CUDA graph replay (step 2+) must not hang")
    public void testFullDecodeReplayDoesNotHang() throws Exception {
        List<Integer> tokens = runDecodeLoop(MAX_STEPS);
        log.info("Generated {} tokens: {}", tokens.size(), tokens);

        // Verify we got meaningful output
        assertEquals(MAX_STEPS, tokens.size(), "Should generate exactly " + MAX_STEPS + " tokens");
        Set<Integer> unique = new HashSet<>(tokens);
        assertTrue(unique.size() >= 3, "Should produce diverse tokens, got " + unique.size());

        // Verify no infinite loops (all same token)
        long maxRepeat = tokens.stream()
                .collect(java.util.stream.Collectors.groupingBy(t -> t, java.util.stream.Collectors.counting()))
                .values().stream().mapToLong(Long::longValue).max().orElse(0);
        assertTrue(maxRepeat < MAX_STEPS * 0.8,
                "Should not repeat same token for >80% of steps (maxRepeat=" + maxRepeat + ")");
    }

    // ========================================================================
    // Test 2: Multiple replay cycles — stress the replay path
    // ========================================================================

    /**
     * Runs 50 decode steps to stress the replay path beyond the initial capture/replay cycle.
     * If pool trimming is releasing memory backing captured graph addresses, this will fail
     * on a later step (not necessarily step 2).
     */
    @Test
    @Order(2)
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    @DisplayName("2. Extended replay — 50 decode steps")
    public void testExtendedReplay() throws Exception {
        List<Integer> tokens = runDecodeLoop(50);
        log.info("Extended: generated {} tokens", tokens.size());
        assertEquals(50, tokens.size(), "Should generate 50 tokens without hanging");
    }

    // ========================================================================
    // Test 3: Back-to-back decode sessions — test resetSession + re-capture
    // ========================================================================

    /**
     * Runs two decode sessions back-to-back. This tests:
     * - resetSession properly cleans up CUDA graph state
     * - Second session can capture and replay without stale state
     * This mimics what happens when processing multiple VLM pages.
     */
    @Test
    @Order(3)
    @Timeout(value = 180, unit = TimeUnit.SECONDS)
    @DisplayName("3. Two decode sessions — reset and re-capture")
    public void testTwoSessionsBackToBack() throws Exception {
        // Session 1
        log.info("=== Session 1 ===");
        List<Integer> tokens1 = runDecodeLoop(15);
        log.info("Session 1: {} tokens", tokens1.size());
        assertEquals(15, tokens1.size());

        // Session 2 — decoder.resetSession() is called inside runDecodeLoop
        log.info("=== Session 2 ===");
        List<Integer> tokens2 = runDecodeLoop(15);
        log.info("Session 2: {} tokens", tokens2.size());
        assertEquals(15, tokens2.size());

        // Both sessions should produce tokens (they won't be identical due to reset)
        Set<Integer> unique1 = new HashSet<>(tokens1);
        Set<Integer> unique2 = new HashSet<>(tokens2);
        assertTrue(unique1.size() >= 2, "Session 1 should produce diverse tokens");
        assertTrue(unique2.size() >= 2, "Session 2 should produce diverse tokens");
    }

    // ========================================================================
    // Test 4: Decode without outputDirect — isolate CUDA graph capture path
    // ========================================================================

    /**
     * Runs decode loop using output() instead of outputDirect() for all steps.
     * This avoids the frozen capture path and tests basic DSP execution.
     * If this passes but test 1 fails, the issue is in the CUDA graph replay mechanism.
     */
    @Test
    @Order(4)
    @Timeout(value = TIMEOUT_SECONDS, unit = TimeUnit.SECONDS)
    @DisplayName("4. Decode without outputDirect — no CUDA graph capture")
    public void testDecodeWithoutOutputDirect() throws Exception {
        long hiddenSize = embeddingTable.size(1);
        int[] promptTokens = {0, 44, 2692, 46};
        INDArray prefillEmbeds = buildPrefillEmbeddings(promptTokens);
        long seqLen = prefillEmbeds.shape()[1];

        decoder.resetSession();
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        ArrayCacheMemoryMgr.setGrowthFactor(1.0);

        Map<String, INDArray> inputs = DecoderUtils.buildDecoderInputMap(
                decoder.inputs(), decoder,
                prefillEmbeds, null,
                0, seqLen, null, 0, 0,
                false, hiddenSize);

        Map<String, INDArray> result = decoder.output(inputs, allOutputNames.toArray(new String[0]));
        int firstToken = result.get(logitsName).get(NDArrayIndex.point(0),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all()).argMax().getInt(0);

        List<Integer> generated = new ArrayList<>();
        generated.add(firstToken);

        long maxKvLen = seqLen + MAX_STEPS;
        Map<String, INDArray> staticKvBuffers = DecoderUtils.padKvCacheToStaticSize(
                result, keyNames, valueNames, maxKvLen);
        Nd4j.getExecutioner().commit();
        long cachePos = seqLen;
        long pastSeqLen = seqLen;

        // Clear prefill DSP plan
        {
            InferenceSession sess = decoder.getOrCreateSession();
            DynamicShapePlanExecutor exec = sess.getDynamicShapePlanExecutor();
            if (exec != null && exec.getCurrentPlan() != null) {
                decoder.clearDynamicShapePlanCache();
                sess.clearAllCaches();
            }
        }

        // ALL steps use output() — no outputDirect, no CUDA graph capture
        for (int step = 1; step < MAX_STEPS; step++) {
            int prevToken = generated.get(generated.size() - 1);
            INDArray tokenEmbed = embeddingTable.getRow(prevToken).reshape(1, 1, hiddenSize);
            INDArray inputIds = Nd4j.createFromArray(new int[]{prevToken}).reshape(1, 1).castTo(DataType.LONG);

            Map<String, INDArray> decInputs = DecoderUtils.buildDecoderInputMap(
                    decoder.inputs(), decoder,
                    tokenEmbed, inputIds,
                    pastSeqLen, 1,
                    staticKvBuffers, maxKvLen, cachePos,
                    true, hiddenSize, null, false);

            result = decoder.output(decInputs, allOutputNames.toArray(new String[0]));

            int nextToken = result.get(logitsName).get(NDArrayIndex.point(0),
                    NDArrayIndex.point(0), NDArrayIndex.all()).argMax().getInt(0);
            generated.add(nextToken);

            DecoderUtils.scatterNewKvEntries(staticKvBuffers, result, keyNames, valueNames, maxKvLen, cachePos);
            cachePos++;
            pastSeqLen++;
        }

        log.info("No-outputDirect: tokens={}", generated);
        assertEquals(MAX_STEPS, generated.size());
        Set<Integer> unique = new HashSet<>(generated);
        assertTrue(unique.size() >= 3, "Should produce diverse tokens, got " + unique.size());
    }

    // ========================================================================
    // Test 5: Decode without DSP — pure baseline
    // ========================================================================

    /**
     * Runs decode without DSP at all. If this hangs, the problem is NOT in DSP/CUDA graph.
     */
    @Test
    @Order(5)
    @Timeout(value = TIMEOUT_SECONDS, unit = TimeUnit.SECONDS)
    @DisplayName("5. Decode without DSP — pure baseline")
    public void testDecodeWithoutDsp() throws Exception {
        long hiddenSize = embeddingTable.size(1);
        int[] promptTokens = {0, 44, 2692, 46};
        INDArray prefillEmbeds = buildPrefillEmbeddings(promptTokens);
        long seqLen = prefillEmbeds.shape()[1];

        decoder.resetSession();
        decoder.setDspAutoCompileEnabled(false);
        decoder.setDspNativeAutoCompileEnabled(false);
        decoder.clearDynamicShapePlanCache();
        ArrayCacheMemoryMgr.setGrowthFactor(1.0);

        Map<String, INDArray> inputs = DecoderUtils.buildDecoderInputMap(
                decoder.inputs(), decoder,
                prefillEmbeds, null,
                0, seqLen, null, 0, 0,
                false, hiddenSize);

        Map<String, INDArray> result = decoder.output(inputs, allOutputNames.toArray(new String[0]));
        int firstToken = result.get(logitsName).get(NDArrayIndex.point(0),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all()).argMax().getInt(0);

        List<Integer> generated = new ArrayList<>();
        generated.add(firstToken);

        long maxKvLen = seqLen + 10;  // Only 10 steps for baseline
        Map<String, INDArray> staticKvBuffers = DecoderUtils.padKvCacheToStaticSize(
                result, keyNames, valueNames, maxKvLen);
        Nd4j.getExecutioner().commit();
        long cachePos = seqLen;
        long pastSeqLen = seqLen;

        for (int step = 1; step < 10; step++) {
            int prevToken = generated.get(generated.size() - 1);
            INDArray tokenEmbed = embeddingTable.getRow(prevToken).reshape(1, 1, hiddenSize);
            INDArray inputIds = Nd4j.createFromArray(new int[]{prevToken}).reshape(1, 1).castTo(DataType.LONG);

            Map<String, INDArray> decInputs = DecoderUtils.buildDecoderInputMap(
                    decoder.inputs(), decoder,
                    tokenEmbed, inputIds,
                    pastSeqLen, 1,
                    staticKvBuffers, maxKvLen, cachePos,
                    true, hiddenSize, null, false);

            result = decoder.output(decInputs, allOutputNames.toArray(new String[0]));
            int nextToken = result.get(logitsName).get(NDArrayIndex.point(0),
                    NDArrayIndex.point(0), NDArrayIndex.all()).argMax().getInt(0);
            generated.add(nextToken);

            DecoderUtils.scatterNewKvEntries(staticKvBuffers, result, keyNames, valueNames, maxKvLen, cachePos);
            cachePos++;
            pastSeqLen++;
        }

        log.info("No-DSP baseline: tokens={}", generated);
        assertEquals(10, generated.size());
    }
}
