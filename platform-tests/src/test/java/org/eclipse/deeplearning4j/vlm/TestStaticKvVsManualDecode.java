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

import static org.junit.jupiter.api.Assertions.*;

/**
 * Compares decode paths using the REAL SmolDocling decoder model.
 * Loads decoder + embed_tokens ONCE. No vision encoder (saves GPU memory).
 *
 * Tests isolate the exact point where DSP/Triton configuration breaks output.
 */
@Slf4j
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@DisplayName("TestStaticKvVsManualDecode")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestStaticKvVsManualDecode {

    private SameDiff decoder;
    private SameDiff embedTokens;
    private INDArray embeddingTable;
    private String logitsName;
    private List<String> kvOutputNames;
    private List<String> allOutputNames;
    private List<String> keyNames;
    private List<String> valueNames;
    private static final int MAX_STEPS = 20;

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
     * Test 0: Single-step prefill comparison — DSP vs non-DSP.
     * Runs the SAME input through the decoder ONCE, with and without DSP.
     * If the output logits differ, the DSP plan compilation or execution is wrong.
     * Also tests with native disabled (Java DSP only) to narrow down.
     */
    @Test
    @Order(0)
    @DisplayName("0. Single-step DSP vs non-DSP comparison")
    public void testSingleStepDspVsNonDsp() {
        long hiddenSize = embeddingTable.size(1);
        int[] promptTokens = {0, 44, 2692, 46};
        INDArray prefillEmbeds = buildPrefillEmbeddings(promptTokens);
        long seqLen = prefillEmbeds.shape()[1];

        // --- Run 1: Non-DSP (baseline) ---
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

        Map<String, INDArray> baselineResult = decoder.output(inputs, logitsName);
        INDArray baselineLogits = baselineResult.get(logitsName);
        int baselineToken = baselineLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all()).argMax().getInt(0);
        INDArray baselineLastLogits = baselineLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all()).dup();
        log.info("Baseline (no DSP): firstToken={} logits[0:5]={}", baselineToken,
                baselineLastLogits.get(NDArrayIndex.interval(0, 5)));

        // Snapshot checksums of ALL constants and variables BEFORE DSP
        Map<String, Double> preChecksums = new LinkedHashMap<>();
        for (SDVariable v : decoder.variables()) {
            if (v.getVariableType() == org.nd4j.autodiff.samediff.VariableType.CONSTANT
                    || v.getVariableType() == org.nd4j.autodiff.samediff.VariableType.VARIABLE) {
                INDArray arr = decoder.getArrForVarName(v.name());
                if (arr != null && !arr.wasClosed() && arr.length() > 0 && !arr.isEmpty()) {
                    try {
                        preChecksums.put(v.name(), arr.sumNumber().doubleValue());
                    } catch (Exception e) {
                        // skip arrays that can't be summed
                    }
                }
            }
        }
        log.info("Pre-DSP: captured {} checksums", preChecksums.size());

        // --- Run 2: Java DSP only (native disabled) ---
        decoder.resetSession();
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(false);
        decoder.clearDynamicShapePlanCache();
        ArrayCacheMemoryMgr.setGrowthFactor(1.0);

        // Log DSP plan info after first execution
        Map<String, INDArray> javaDspResult = decoder.output(inputs, logitsName);
        {
            InferenceSession sess = decoder.getOrCreateSession();
            DynamicShapePlanExecutor exec = sess.getDynamicShapePlanExecutor();
            if (exec != null && exec.getCurrentPlan() != null) {
                var plan = exec.getCurrentPlan();
                log.info("DSP plan: slots={} extInputs={} outputs={}",
                        plan.getSlots().length,
                        plan.getExternalInputKeys().length,
                        plan.getRequestedOutputs());
                // Check a few external inputs
                String[] extKeys = plan.getExternalInputKeys();
                byte[] extTypes = plan.getExternalInputSourceTypes();
                int phCount = 0, constCount = 0, varCount = 0;
                for (int i = 0; i < extTypes.length; i++) {
                    switch (extTypes[i]) {
                        case 0: constCount++; break; // SOURCE_CONSTANT
                        case 1: varCount++; break;   // SOURCE_VARIABLE
                        case 2: phCount++; break;    // SOURCE_PLACEHOLDER
                    }
                }
                log.info("External inputs: {} constants, {} variables, {} placeholders",
                        constCount, varCount, phCount);
                // Log placeholder names
                for (int i = 0; i < extKeys.length; i++) {
                    if (extTypes[i] == 2) { // SOURCE_PLACEHOLDER
                        log.info("  Placeholder[{}]: '{}'", i, extKeys[i]);
                    }
                }
            }
        }
        INDArray javaDspLogits = javaDspResult.get(logitsName);
        int javaDspToken = javaDspLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all()).argMax().getInt(0);
        INDArray javaDspLastLogits = javaDspLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all()).dup();
        log.info("Java DSP: firstToken={} logits[0:5]={}", javaDspToken,
                javaDspLastLogits.get(NDArrayIndex.interval(0, 5)));

        double javaDspDiff = javaDspLastLogits.sub(baselineLastLogits).amaxNumber().doubleValue();
        log.info("Java DSP vs baseline: maxDiff={}", javaDspDiff);

        // Checksum comparison: find which constants/variables were corrupted by DSP
        int corrupted = 0;
        for (Map.Entry<String, Double> entry : preChecksums.entrySet()) {
            String name = entry.getKey();
            double preSumVal = entry.getValue();
            INDArray arr = decoder.getArrForVarName(name);
            if (arr != null && !arr.wasClosed() && arr.length() > 0 && !arr.isEmpty()) {
                try {
                    double postSum = arr.sumNumber().doubleValue();
                    double diff = Math.abs(postSum - preSumVal);
                    if (diff > 1e-6) {
                        SDVariable sdv = decoder.getVariable(name);
                        log.error("CORRUPTED: '{}' type={} shape={} preSum={} postSum={} diff={}",
                                name, sdv != null ? sdv.getVariableType() : "?",
                                Arrays.toString(arr.shape()), preSumVal, postSum, diff);
                        corrupted++;
                    }
                } catch (Exception e) {
                    // skip
                }
            } else {
                log.warn("POST-DSP: '{}' is null/closed/empty", name);
            }
        }
        log.info("Checksum comparison: {} of {} constants/variables corrupted", corrupted, preChecksums.size());

        // Try running non-DSP on the SAME session (without resetSession) to check contamination
        decoder.setDspAutoCompileEnabled(false);
        Map<String, INDArray> postDspNoReset = decoder.output(inputs, logitsName);
        int postDspNoResetToken = postDspNoReset.get(logitsName).get(NDArrayIndex.point(0),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all()).argMax().getInt(0);
        log.info("Post-DSP (no reset, DSP off): token={}", postDspNoResetToken);

        assertEquals(baselineToken, javaDspToken,
                "Java DSP should produce same token as baseline. Baseline=" + baselineToken + " JavaDSP=" + javaDspToken);

        // --- Run 3: Native DSP (Java + native) ---
        decoder.resetSession();
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        decoder.clearDynamicShapePlanCache();
        ArrayCacheMemoryMgr.setGrowthFactor(1.0);

        Map<String, INDArray> nativeDspResult = decoder.output(inputs, logitsName);
        INDArray nativeDspLogits = nativeDspResult.get(logitsName);
        int nativeDspToken = nativeDspLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all()).argMax().getInt(0);
        INDArray nativeDspLastLogits = nativeDspLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all()).dup();
        log.info("Native DSP: firstToken={} logits[0:5]={}", nativeDspToken,
                nativeDspLastLogits.get(NDArrayIndex.interval(0, 5)));

        double nativeDspDiff = nativeDspLastLogits.sub(baselineLastLogits).amaxNumber().doubleValue();
        log.info("Native DSP vs baseline: maxDiff={}", nativeDspDiff);
        assertEquals(baselineToken, nativeDspToken,
                "Native DSP should produce same token as baseline. Baseline=" + baselineToken + " NativeDSP=" + nativeDspToken);

        // --- Run 4: Non-DSP again (confirm no contamination) ---
        decoder.resetSession();
        decoder.setDspAutoCompileEnabled(false);
        decoder.setDspNativeAutoCompileEnabled(false);
        decoder.clearDynamicShapePlanCache();
        ArrayCacheMemoryMgr.setGrowthFactor(1.0);

        Map<String, INDArray> cleanResult = decoder.output(inputs, logitsName);
        int cleanToken = cleanResult.get(logitsName).get(NDArrayIndex.point(0),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all()).argMax().getInt(0);
        log.info("Clean baseline (post-DSP): firstToken={}", cleanToken);
        assertEquals(baselineToken, cleanToken,
                "Post-DSP baseline should still produce same token. Baseline=" + baselineToken + " Clean=" + cleanToken);

        log.info("SUMMARY: baseline={} javaDsp={} nativeDsp={} clean={}",
                baselineToken, javaDspToken, nativeDspToken, cleanToken);
    }

    /**
     * Test 1: Static KV decode WITHOUT DSP — baseline correctness.
     * Uses buildDecoderInputMap + scatter, growing views, all output().
     * This matches what we already confirmed works.
     */
    @Test
    @Order(1)
    @DisplayName("1. Static KV decode (no DSP) — baseline")
    public void testStaticKvNoDsp() {
        long hiddenSize = embeddingTable.size(1);
        int[] promptTokens = {0, 44, 2692, 46};
        INDArray prefillEmbeds = buildPrefillEmbeddings(promptTokens);
        long seqLen = prefillEmbeds.shape()[1];

        // Prefill
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

        log.info("Baseline (no DSP): tokens={}", generated);
        Set<Integer> unique = new HashSet<>(generated);
        log.info("Unique: {}/{}", unique.size(), generated.size());
        assertTrue(unique.size() >= 3, "Baseline should produce some diverse tokens, got " + unique.size());
    }

    /**
     * Test 2: Static KV decode WITH DSP auto-compile enabled.
     * DSP will compile a plan after step 1, then recompile every step (growing shapes).
     * If output diverges from test 1, DSP recompilation is the bug.
     */
    @Test
    @Order(2)
    @DisplayName("2. Static KV decode WITH DSP — growing shapes")
    public void testStaticKvWithDsp() {
        long hiddenSize = embeddingTable.size(1);
        int[] promptTokens = {0, 44, 2692, 46};
        INDArray prefillEmbeds = buildPrefillEmbeddings(promptTokens);
        long seqLen = prefillEmbeds.shape()[1];

        decoder.resetSession();

        // Enable DSP
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);

        // Prefill
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

        // Clear stale prefill DSP plan (like StaticKvCacheDecodeLoop does)
        {
            InferenceSession sess = decoder.getOrCreateSession();
            DynamicShapePlanExecutor exec = sess.getDynamicShapePlanExecutor();
            if (exec != null && exec.getCurrentPlan() != null) {
                decoder.clearDynamicShapePlanCache();
                sess.clearAllCaches();
                log.info("Cleared stale prefill DSP plan");
            }
        }

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

            // Log DSP state
            InferenceSession sess = decoder.getOrCreateSession();
            DynamicShapePlanExecutor exec = sess.getDynamicShapePlanExecutor();
            boolean hasPlan = exec != null && exec.getCurrentPlan() != null;
            if (step <= 5) {
                log.info("DSP step {}: token={} dspPlan={}", step, nextToken, hasPlan);
            }

            DecoderUtils.scatterNewKvEntries(staticKvBuffers, result, keyNames, valueNames, maxKvLen, cachePos);
            cachePos++;
            pastSeqLen++;
        }

        log.info("DSP enabled: tokens={}", generated);
        Set<Integer> unique = new HashSet<>(generated);
        log.info("Unique: {}/{}", unique.size(), generated.size());
        assertTrue(unique.size() >= 3, "DSP decode should produce diverse tokens, got " + unique.size());

        // Disable DSP for subsequent tests
        decoder.setDspAutoCompileEnabled(false);
        decoder.setDspNativeAutoCompileEnabled(false);
    }

    /**
     * Test 3: Static KV decode with outputDirect (step >= 2).
     * No DSP. Tests whether outputDirect itself causes issues.
     */
    @Test
    @Order(3)
    @DisplayName("3. Static KV + outputDirect (no DSP)")
    public void testStaticKvOutputDirect() {
        long hiddenSize = embeddingTable.size(1);
        int[] promptTokens = {0, 44, 2692, 46};
        INDArray prefillEmbeds = buildPrefillEmbeddings(promptTokens);
        long seqLen = prefillEmbeds.shape()[1];

        decoder.resetSession();

        // Prefill
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

            // outputDirect from step 2 — like StaticKvCacheDecodeLoop
            boolean useDirect = step >= 2;
            if (useDirect) {
                result = decoder.outputDirect(decInputs, allOutputNames.toArray(new String[0]));
            } else {
                result = decoder.output(decInputs, allOutputNames.toArray(new String[0]));
            }

            int nextToken = result.get(logitsName).get(NDArrayIndex.point(0),
                    NDArrayIndex.point(0), NDArrayIndex.all()).argMax().getInt(0);
            generated.add(nextToken);

            DecoderUtils.scatterNewKvEntries(staticKvBuffers, result, keyNames, valueNames, maxKvLen, cachePos);
            cachePos++;
            pastSeqLen++;
        }

        log.info("outputDirect: tokens={}", generated);
        Set<Integer> unique = new HashSet<>(generated);
        log.info("Unique: {}/{}", unique.size(), generated.size());
        assertTrue(unique.size() >= 3, "outputDirect should produce diverse tokens, got " + unique.size());
    }

    /**
     * Test 4: DSP + outputDirect + KV close (full StaticKvCacheDecodeLoop path).
     * This is the EXACT production code path. If this fails, we've found the bug.
     */
    @Test
    @Order(4)
    @DisplayName("4. Full decode loop path (DSP + outputDirect + KV close)")
    public void testFullDecodeLoopPath() {
        long hiddenSize = embeddingTable.size(1);
        int[] promptTokens = {0, 44, 2692, 46};
        INDArray prefillEmbeds = buildPrefillEmbeddings(promptTokens);
        long seqLen = prefillEmbeds.shape()[1];

        decoder.resetSession();
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);

        // Prefill
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

            // outputDirect from step 2
            boolean useDirect = step >= 2;
            if (useDirect) {
                result = decoder.outputDirect(decInputs, allOutputNames.toArray(new String[0]));
            } else {
                result = decoder.output(decInputs, allOutputNames.toArray(new String[0]));
            }

            int nextToken = result.get(logitsName).get(NDArrayIndex.point(0),
                    NDArrayIndex.point(0), NDArrayIndex.all()).argMax().getInt(0);
            generated.add(nextToken);

            // Scatter
            DecoderUtils.scatterNewKvEntries(staticKvBuffers, result, keyNames, valueNames, maxKvLen, cachePos);
            cachePos++;

            // Close present KV outputs (like StaticKvCacheDecodeLoop does)
            boolean nativePlanFrozen = false;
            {
                InferenceSession sess = decoder.getOrCreateSession();
                DynamicShapePlanExecutor exec = sess.getDynamicShapePlanExecutor();
                nativePlanFrozen = exec != null && exec.isShapesFrozen();
            }
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

            if (step <= 5) {
                log.info("Full path step {}: token={}", step, nextToken);
            }
        }

        log.info("Full path: tokens={}", generated);
        Set<Integer> unique = new HashSet<>(generated);
        log.info("Unique: {}/{}", unique.size(), generated.size());
        assertTrue(unique.size() >= 3, "Full path should produce diverse tokens, got " + unique.size());

        decoder.setDspAutoCompileEnabled(false);
        decoder.setDspNativeAutoCompileEnabled(false);
    }
}
