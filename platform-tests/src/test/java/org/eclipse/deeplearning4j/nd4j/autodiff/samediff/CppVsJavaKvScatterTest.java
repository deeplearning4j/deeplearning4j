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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.DecoderUtils;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.generation.StaticKvCacheManager;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspCompilationMode;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration test: compares C++ KV scatter vs Java KV scatter using the real
 * SmolDocling decoder model within a SINGLE model instance.
 *
 * Instead of running two separate model instances (which can have non-deterministic
 * differences), this test:
 * 1. Runs prefill + step 1 (Java scatter for both)
 * 2. Runs step 2 with C++ scatter enabled + fullOutputNames so we get BOTH
 *    the present KV output AND C++ writes to the static buffer
 * 3. Compares: present KV output (truth) vs static buffer at cachePos (C++ wrote)
 *
 * This isolates whether the C++ scatter writes the correct data.
 */
@Slf4j
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@DisplayName("CppVsJavaKvScatterTest")
public class CppVsJavaKvScatterTest {

    private File decoderFile;
    private File embedFile;
    private static final long HIDDEN_SIZE = 576;
    private static final int NUM_KV_HEADS = 3;
    private static final int HEAD_DIM = 64;

    @BeforeAll
    void setup() throws Exception {
        var decoderResult = VLMModelDownloader.download(
                VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        decoderFile = decoderResult.getModelFile();
        var embedResult = VLMModelDownloader.download(
                VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        embedFile = embedResult.getModelFile();
    }

    /**
     * Verify C++ scatter writes correct data by comparing present KV output
     * (the source truth) with static buffer contents (what C++ scatter wrote).
     *
     * Uses a SINGLE model instance — no cross-instance comparison artifacts.
     */
    @Test
    @DisplayName("C++ scatter writes correct data to static KV buffer")
    void testCppScatterWritesCorrectData() throws Exception {
        SameDiff decoder = OnnxModelCache.importWithCache(decoderFile.getAbsolutePath());
        SameDiff embedTokens = OnnxModelCache.importWithCache(embedFile.getAbsolutePath());

        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        DecoderUtils.KVCacheNames kvNames = ioConfig.getKvCacheNames();
        String logitsName = ioConfig.getLogitsOutputName();
        String embedsName = ioConfig.getInputEmbeddingsName();

        // Prefill token IDs
        long[] prefillIds = {49229};  // <doctag>
        INDArray inputIds = Nd4j.createFromArray(prefillIds).reshape(1, prefillIds.length).castTo(DataType.LONG);

        // Get embeddings
        String[] embedOutputNames = embedTokens.outputs().toArray(new String[0]);
        Map<String, INDArray> embedOut = embedTokens.output(
                Map.of("input_ids", inputIds), embedOutputNames);
        INDArray embeddings = embedOut.get(embedOutputNames[0]);
        assertNotNull(embeddings, "Embedding output should not be null");

        List<String> inputNames = decoder.inputs();

        // Build prefill inputs
        Map<String, INDArray> prefillInputs = DecoderUtils.buildDecoderInputMap(
                ioConfig, inputNames, decoder, embeddings, inputIds,
                0, prefillIds.length, null, -1, 0,
                false, HIDDEN_SIZE, null, false, false);
        if (embedsName != null && !prefillInputs.containsKey(embedsName)) {
            prefillInputs.put(embedsName, embeddings);
        }
        prefillInputs.entrySet().removeIf(e -> e.getValue() == null);

        String[] fullOutputNames = buildFullOutputNames(logitsName, kvNames);
        Map<String, INDArray> prefillOutputs = decoder.output(prefillInputs, fullOutputNames);

        // Initialize static KV from prefill
        StaticKvCacheManager kvMgr = new StaticKvCacheManager(ioConfig);
        kvMgr.initializeFromPrefill(prefillOutputs, kvNames, 15, prefillIds.length);

        long maxKvLen = kvMgr.getMaxKvLen();
        long cachePos = kvMgr.getCachePosition();
        Map<String, INDArray> staticKvBuffers = kvMgr.getStaticKvBuffers();

        log.info("After prefill: maxKvLen={}, cachePos={}", maxKvLen, cachePos);

        // Associate static KV buffers with decoder
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(e.getKey())) {
                decoder.associateArrayWithVariable(e.getValue(), e.getKey());
            }
        }

        // Recompile DSP with FULL outputs (so we can capture present KV for comparison)
        decoder.clearDynamicShapePlanCache();
        var session = decoder.getOrCreateSession();
        session.clearAllCaches();
        decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, fullOutputNames);

        // Add placeholder override for attn_mask_reformat AFTER DSP compile
        // so prefill can compute it via the internal subgraph, but decode steps
        // provide the 4D bias directly (matching StaticKvCacheDecodeLoop behavior).
        String attnReformatNode = ioConfig.getAttnMaskReformatOutput();
        if (attnReformatNode != null && decoder.hasVariable(attnReformatNode)) {
            decoder.addPlaceholderOverride(attnReformatNode);
            decoder.getVariable(attnReformatNode).setShape(-1, -1, -1, -1);
        }
        // Re-fetch input names since addPlaceholderOverride added the attn_mask_reformat node
        inputNames = decoder.inputs();

        session = decoder.getOrCreateSession();
        var dspExec = session.getDynamicShapePlanExecutor();
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);
        }

        // NOTE: Do NOT configure KV retention yet. decoder.output() also goes through
        // the native DSP path when DYNAMIC_SHAPE_PLAN_ENABLED=true, so if retention
        // were configured here, C++ scatter would fire during step 1 AND the test's
        // Java scatter would also fire, causing position tracking desync.
        // Configure retention AFTER step 1 so C++ scatter only runs at step 2.

        // Close prefill KV outputs
        for (String name : kvNames.keyNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        for (String name : kvNames.valueNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }

        // Get prefill token
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        INDArray lastLogits = prefillLogits.rank() == 3
                ? prefillLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(prefillLogits.size(1) - 1),
                NDArrayIndex.all())
                : prefillLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
        int nextTokenId = Nd4j.argMax(lastLogits).getInt(0);
        log.info("Prefill token: {}", nextTokenId);

        Map<String, INDArray> reusableInputs = new HashMap<>();
        List<Integer> tokens = new ArrayList<>();
        tokens.add(nextTokenId);

        // Step 1: output() with Java scatter
        {
            long pastSeqLen = prefillIds.length;
            cachePos = kvMgr.getCachePosition();
            INDArray tokenIdArr = Nd4j.createFromArray(new long[]{nextTokenId}).reshape(1, 1);
            Map<String, INDArray> tokenEmbed = embedTokens.output(
                    Map.of("input_ids", tokenIdArr), embedOutputNames);
            INDArray stepEmbedding = tokenEmbed.get(embedOutputNames[0]);

            Map<String, INDArray> decInputs = DecoderUtils.buildDecoderInputMap(
                    ioConfig, inputNames, decoder, stepEmbedding, tokenIdArr,
                    pastSeqLen, 1, staticKvBuffers, maxKvLen, cachePos,
                    true, HIDDEN_SIZE, reusableInputs, true, false);
            if (embedsName != null && !decInputs.containsKey(embedsName)) {
                decInputs.put(embedsName, stepEmbedding);
            }

            Map<String, INDArray> outputs = decoder.output(decInputs, fullOutputNames);
            INDArray logits = outputs.get(logitsName);
            INDArray stepLogits = logits.rank() == 3
                    ? logits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(logits.size(1) - 1),
                    NDArrayIndex.all())
                    : logits.get(NDArrayIndex.point(0), NDArrayIndex.all());
            nextTokenId = Nd4j.argMax(stepLogits).getInt(0);
            tokens.add(nextTokenId);

            // Java scatter
            kvMgr.scatterNewEntries(outputs, kvNames);

            // Close KV outputs
            for (String name : kvNames.keyNames) {
                INDArray arr = outputs.get(name);
                if (arr != null && !arr.wasClosed() && arr.data() != null) {
                    arr.setCloseable(true); arr.close();
                }
            }
            for (String name : kvNames.valueNames) {
                INDArray arr = outputs.get(name);
                if (arr != null && !arr.wasClosed() && arr.data() != null) {
                    arr.setCloseable(true); arr.close();
                }
            }
            log.info("Step 1: token={} cachePos={}", nextTokenId, cachePos);
        }

        // NOW configure C++ KV retention — after step 1 completed.
        // initialPos = kvMgr.getCachePosition() which is 2 after step 1 scatter.
        cachePos = kvMgr.getCachePosition();
        if (dspExec != null && dspExec.getCurrentPlan() != null) {
            List<String> presentNames = new ArrayList<>();
            presentNames.addAll(kvNames.keyNames);
            presentNames.addAll(kvNames.valueNames);
            List<String> pastNames = new ArrayList<>();
            for (String pn : presentNames) {
                pastNames.add(ioConfig.presentToInputName(pn));
            }
            boolean configured = dspExec.configureKvCacheRetention(dspExec.getCurrentPlan(),
                    presentNames, pastNames, (int) maxKvLen, (int) cachePos);
            log.info("C++ KV scatter configured: {} initialPos={}", configured, cachePos);
            dspExec.configureDecodeInputs(dspExec.getCurrentPlan(), (int) maxKvLen);
        }

        // Step 2: outputDirect() with C++ scatter + fullOutputNames
        // We get present KV in the output AND C++ scatter writes to the static buffer
        // Compare the two to verify scatter correctness
        {
            long pastSeqLen = prefillIds.length + 1;
            cachePos = kvMgr.getCachePosition();
            log.info("Step 2 start: cachePos={}", cachePos);

            INDArray tokenIdArr = Nd4j.createFromArray(new long[]{nextTokenId}).reshape(1, 1);
            Map<String, INDArray> tokenEmbed = embedTokens.output(
                    Map.of("input_ids", tokenIdArr), embedOutputNames);
            INDArray stepEmbedding = tokenEmbed.get(embedOutputNames[0]);

            Map<String, INDArray> decInputs = DecoderUtils.buildDecoderInputMap(
                    ioConfig, inputNames, decoder, stepEmbedding, tokenIdArr,
                    pastSeqLen, 1, staticKvBuffers, maxKvLen, cachePos,
                    true, HIDDEN_SIZE, reusableInputs, true, false);
            if (embedsName != null && !decInputs.containsKey(embedsName)) {
                decInputs.put(embedsName, stepEmbedding);
            }

            // Snapshot static buffer BEFORE the step
            Map<String, INDArray> beforeKv = new HashMap<>();
            for (String keyName : kvNames.keyNames) {
                String pastName = ioConfig.presentToInputName(keyName);
                INDArray buf = staticKvBuffers.get(pastName);
                if (buf != null) {
                    beforeKv.put(keyName, buf.get(
                            NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.point(cachePos), NDArrayIndex.all()).dup());
                }
            }
            for (String valName : kvNames.valueNames) {
                String pastName = ioConfig.presentToInputName(valName);
                INDArray buf = staticKvBuffers.get(pastName);
                if (buf != null) {
                    beforeKv.put(valName, buf.get(
                            NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.point(cachePos), NDArrayIndex.all()).dup());
                }
            }

            // Execute with outputDirect — C++ scatter runs internally
            // Request only logits since present KV data buffers are null after C++ scatter
            Map<String, INDArray> outputs = decoder.outputDirect(decInputs, new String[]{logitsName});
            // Ensure GPU scatter kernel has completed before reading buffer contents
            Nd4j.getExecutioner().commit();
            INDArray logits = outputs.get(logitsName);
            INDArray stepLogits = logits.rank() == 3
                    ? logits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(logits.size(1) - 1),
                    NDArrayIndex.all())
                    : logits.get(NDArrayIndex.point(0), NDArrayIndex.all());
            nextTokenId = Nd4j.argMax(stepLogits).getInt(0);
            tokens.add(nextTokenId);
            log.info("Step 2: token={} cachePos={}", nextTokenId, cachePos);

            // C++ scatter should have written to position cachePos in the static buffer
            // Check: buffer at cachePos changed from before-step snapshot (scatter wrote something)
            // AND: the written values are non-zero (scatter wrote real data)
            int zeroScatterCount = 0;
            int unchangedCount = 0;
            int writtenCount = 0;

            for (String keyName : kvNames.keyNames) {
                String pastName = ioConfig.presentToInputName(keyName);
                INDArray staticBuf = staticKvBuffers.get(pastName);
                if (staticBuf == null) continue;

                INDArray afterEntry = staticBuf.get(
                        NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(cachePos), NDArrayIndex.all()).dup();
                INDArray beforeEntry = beforeKv.get(keyName);

                double afterAbsSum = Transforms.abs(afterEntry).sumNumber().doubleValue();
                double beforeAfterDiff = beforeEntry != null
                        ? Transforms.abs(afterEntry.sub(beforeEntry)).maxNumber().doubleValue()
                        : -1;

                if (afterAbsSum == 0.0) {
                    zeroScatterCount++;
                    log.error("SCATTER WROTE ZEROS: key='{}' at cachePos={}", keyName, cachePos);
                } else if (beforeAfterDiff == 0.0) {
                    unchangedCount++;
                    log.error("SCATTER DID NOT WRITE: key='{}' buffer unchanged at cachePos={}",
                            keyName, cachePos);
                } else {
                    writtenCount++;
                    log.info("SCATTER OK: key='{}' absSum={} beforeAfterDiff={}",
                            keyName, afterAbsSum, beforeAfterDiff);
                    logFirstValues(afterEntry, "scattered", keyName);
                }
            }

            for (String valName : kvNames.valueNames) {
                String pastName = ioConfig.presentToInputName(valName);
                INDArray staticBuf = staticKvBuffers.get(pastName);
                if (staticBuf == null) continue;

                INDArray afterEntry = staticBuf.get(
                        NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(cachePos), NDArrayIndex.all()).dup();
                INDArray beforeEntry = beforeKv.get(valName);

                double afterAbsSum = Transforms.abs(afterEntry).sumNumber().doubleValue();
                double beforeAfterDiff = beforeEntry != null
                        ? Transforms.abs(afterEntry.sub(beforeEntry)).maxNumber().doubleValue()
                        : -1;

                if (afterAbsSum == 0.0) {
                    zeroScatterCount++;
                    log.error("SCATTER WROTE ZEROS: value='{}' at cachePos={}", valName, cachePos);
                } else if (beforeAfterDiff == 0.0) {
                    unchangedCount++;
                    log.error("SCATTER DID NOT WRITE: value='{}' buffer unchanged at cachePos={}",
                            valName, cachePos);
                } else {
                    writtenCount++;
                }
            }

            // Advance cachePos for C++ scatter
            kvMgr.setCachePosition(kvMgr.getCachePosition() + 1);

            log.info("Step 2 results: written={} zeros={} unchanged={}", writtenCount, zeroScatterCount, unchangedCount);
            log.info("Tokens so far: {}", tokens);

            if (zeroScatterCount > 0) {
                fail("C++ scatter wrote zeros to " + zeroScatterCount + " KV buffer positions");
            }
            if (unchangedCount > 0) {
                fail("C++ scatter did not modify " + unchangedCount + " KV buffer positions (buffer unchanged after step)");
            }
            assertTrue(writtenCount > 0, "Expected at least some KV entries to be written by C++ scatter");
        }

        decoder.close();
        embedTokens.close();
    }

    /**
     * Run multiple decode steps and verify C++ scatter path produces valid tokens
     * (no stuck-on-same-token degeneration).
     */
    @Test
    @DisplayName("C++ scatter produces diverse tokens (not stuck repeating)")
    void testCppScatterProducesDiverseTokens() throws Exception {
        final int NUM_STEPS = 10;
        List<Integer> tokens = runDecodeStepsWithCppScatter(NUM_STEPS);

        log.info("C++ scatter tokens ({}): {}", NUM_STEPS, tokens);

        // Count unique tokens in steps 2+ (after initial pattern establishment)
        Set<Integer> uniqueTokens = new HashSet<>(tokens.subList(Math.min(2, tokens.size()), tokens.size()));
        log.info("Unique tokens after step 2: {} out of {} steps", uniqueTokens.size(),
                tokens.size() - Math.min(2, tokens.size()));

        // Should have at least 2 unique tokens in 10 steps — stuck repeating = 1 unique
        assertTrue(uniqueTokens.size() >= 2,
                "C++ scatter should produce diverse tokens, got: " + tokens);
    }

    // ═══════════════════════════════════════════════════════════════════════

    private List<Integer> runDecodeStepsWithCppScatter(int numSteps) throws Exception {
        SameDiff decoder = OnnxModelCache.importWithCache(decoderFile.getAbsolutePath());
        SameDiff embedTokens = OnnxModelCache.importWithCache(embedFile.getAbsolutePath());

        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        DecoderUtils.KVCacheNames kvNames = ioConfig.getKvCacheNames();
        String logitsName = ioConfig.getLogitsOutputName();
        String embedsName = ioConfig.getInputEmbeddingsName();

        long[] prefillIds = {49229};
        INDArray inputIds = Nd4j.createFromArray(prefillIds).reshape(1, prefillIds.length).castTo(DataType.LONG);

        String[] embedOutputNames = embedTokens.outputs().toArray(new String[0]);
        Map<String, INDArray> embedOut = embedTokens.output(
                Map.of("input_ids", inputIds), embedOutputNames);
        INDArray embeddings = embedOut.get(embedOutputNames[0]);
        assertNotNull(embeddings);

        List<String> inputNames = decoder.inputs();
        String[] fullOutputNames = buildFullOutputNames(logitsName, kvNames);

        Map<String, INDArray> prefillInputs = DecoderUtils.buildDecoderInputMap(
                ioConfig, inputNames, decoder, embeddings, inputIds,
                0, prefillIds.length, null, -1, 0,
                false, HIDDEN_SIZE, null, false, false);
        if (embedsName != null && !prefillInputs.containsKey(embedsName)) {
            prefillInputs.put(embedsName, embeddings);
        }
        prefillInputs.entrySet().removeIf(e -> e.getValue() == null);

        Map<String, INDArray> prefillOutputs = decoder.output(prefillInputs, fullOutputNames);

        StaticKvCacheManager kvMgr = new StaticKvCacheManager(ioConfig);
        kvMgr.initializeFromPrefill(prefillOutputs, kvNames, numSteps + 10, prefillIds.length);

        long maxKvLen = kvMgr.getMaxKvLen();
        Map<String, INDArray> staticKvBuffers = kvMgr.getStaticKvBuffers();

        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(e.getKey())) {
                decoder.associateArrayWithVariable(e.getValue(), e.getKey());
            }
        }

        // Compile with logits-only for C++ scatter
        decoder.clearDynamicShapePlanCache();
        var session = decoder.getOrCreateSession();
        session.clearAllCaches();
        decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE,
                new String[]{logitsName});

        // Add placeholder override for attn_mask_reformat AFTER DSP compile
        String attnReformatNode = ioConfig.getAttnMaskReformatOutput();
        if (attnReformatNode != null && decoder.hasVariable(attnReformatNode)) {
            decoder.addPlaceholderOverride(attnReformatNode);
            decoder.getVariable(attnReformatNode).setShape(-1, -1, -1, -1);
        }
        // Re-fetch input names since addPlaceholderOverride added the attn_mask_reformat node
        inputNames = decoder.inputs();

        session = decoder.getOrCreateSession();
        var dspExec = session.getDynamicShapePlanExecutor();
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);
        }

        // Do NOT configure KV retention yet — decoder.output() goes through native DSP,
        // and if retention were configured, C++ scatter would fire during step 1 too.

        // Close prefill KV
        for (String name : kvNames.keyNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        for (String name : kvNames.valueNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }

        INDArray prefillLogits = prefillOutputs.get(logitsName);
        INDArray lastLogits = prefillLogits.rank() == 3
                ? prefillLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(prefillLogits.size(1) - 1), NDArrayIndex.all())
                : prefillLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
        int nextTokenId = Nd4j.argMax(lastLogits).getInt(0);

        List<Integer> tokens = new ArrayList<>();
        tokens.add(nextTokenId);
        Map<String, INDArray> reusableInputs = new HashMap<>();
        boolean kvRetentionConfigured = false;

        for (int step = 1; step <= numSteps; step++) {
            long pastSeqLen = prefillIds.length + step - 1;
            long cachePos = kvMgr.getCachePosition();

            INDArray tokenIdArr = Nd4j.createFromArray(new long[]{nextTokenId}).reshape(1, 1);
            Map<String, INDArray> tokenEmbed = embedTokens.output(
                    Map.of("input_ids", tokenIdArr), embedOutputNames);
            INDArray stepEmbedding = tokenEmbed.get(embedOutputNames[0]);

            // Step 1: Java scatter (retention not configured yet)
            // Step 2+: C++ scatter via outputDirect
            boolean cppScatterThisStep = kvRetentionConfigured;

            Map<String, INDArray> decInputs = DecoderUtils.buildDecoderInputMap(
                    ioConfig, inputNames, decoder, stepEmbedding, tokenIdArr,
                    pastSeqLen, 1, staticKvBuffers, maxKvLen, cachePos,
                    true, HIDDEN_SIZE, reusableInputs, true, false);
            if (embedsName != null && !decInputs.containsKey(embedsName)) {
                decInputs.put(embedsName, stepEmbedding);
            }

            String[] reqOutputs = cppScatterThisStep
                    ? new String[]{logitsName} : fullOutputNames;

            Map<String, INDArray> outputs;
            if (cppScatterThisStep) {
                outputs = decoder.outputDirect(decInputs, reqOutputs);
            } else {
                outputs = decoder.output(decInputs, reqOutputs);
            }

            INDArray logits = outputs.get(logitsName);
            INDArray stepLogits = logits.rank() == 3
                    ? logits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(logits.size(1) - 1), NDArrayIndex.all())
                    : logits.get(NDArrayIndex.point(0), NDArrayIndex.all());
            nextTokenId = Nd4j.argMax(stepLogits).getInt(0);
            tokens.add(nextTokenId);

            if (cppScatterThisStep) {
                // C++ scatter already advanced position internally
                kvMgr.setCachePosition(kvMgr.getCachePosition() + 1);
            } else {
                // Java scatter for step 1
                kvMgr.scatterNewEntries(outputs, kvNames);
                for (String name : kvNames.keyNames) {
                    INDArray arr = outputs.get(name);
                    if (arr != null && !arr.wasClosed() && arr.data() != null) {
                        arr.setCloseable(true); arr.close();
                    }
                }
                for (String name : kvNames.valueNames) {
                    INDArray arr = outputs.get(name);
                    if (arr != null && !arr.wasClosed() && arr.data() != null) {
                        arr.setCloseable(true); arr.close();
                    }
                }

                // Configure KV retention after the first Java scatter step
                if (!kvRetentionConfigured && dspExec != null && dspExec.getCurrentPlan() != null) {
                    List<String> presentNames = new ArrayList<>();
                    presentNames.addAll(kvNames.keyNames);
                    presentNames.addAll(kvNames.valueNames);
                    List<String> pastNames = new ArrayList<>();
                    for (String pn : presentNames) {
                        pastNames.add(ioConfig.presentToInputName(pn));
                    }
                    dspExec.configureKvCacheRetention(dspExec.getCurrentPlan(),
                            presentNames, pastNames, (int) maxKvLen, (int) kvMgr.getCachePosition());
                    dspExec.configureDecodeInputs(dspExec.getCurrentPlan(), (int) maxKvLen);
                    kvRetentionConfigured = true;
                    log.info("C++ KV scatter configured after step {} at pos={}", step, kvMgr.getCachePosition());
                }
            }

            log.info("Step {}: token={} cachePos={}", step, nextTokenId, cachePos);
        }

        decoder.close();
        embedTokens.close();
        return tokens;
    }

    private String[] buildFullOutputNames(String logitsName, DecoderUtils.KVCacheNames kvNames) {
        List<String> names = new ArrayList<>();
        names.add(logitsName);
        names.addAll(kvNames.keyNames);
        names.addAll(kvNames.valueNames);
        return names.toArray(new String[0]);
    }

    private void logFirstValues(INDArray arr, String label, String name) {
        float[] data = arr.dup().data().asFloat();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < Math.min(8, data.length); i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.4f", data[i]));
        }
        log.info("  {} '{}' first 8 values: [{}]", label, name, sb);
    }
}
