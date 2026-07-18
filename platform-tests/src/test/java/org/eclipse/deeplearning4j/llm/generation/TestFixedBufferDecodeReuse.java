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

package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.LongPointer;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.DownloadResult;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.LLMModel;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.QuantType;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SameDiffSerializer;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression coverage for the fixed-buffer decode REUSE forward-fix (ADR 0105 follow-up): a
 * {@link GenerationPipeline} configured with {@code maxPrefillLength}/{@code maxKvCacheLength}
 * (the "fixed-buffer fast path") caches its {@link InGraphKvState} across one-shot {@code generate()}
 * calls and reuses the frozen DSP plan + captured CUDA graph + stable-address KV / recurrent /
 * prefill buffers instead of re-warming every generate. This is the scenario no pre-existing test
 * covered (per the DSP/decode test survey): many generates on ONE fixed-buffer pipeline.
 *
 * <p>The fix must be behavior-preserving. This class pins three invariants:</p>
 * <ol>
 *   <li><b>reuse == fresh</b> — generate 0 builds the state fresh (no reuse); generates 1..N reuse it.
 *       Under greedy decoding on the same prompt every reused generate must be token-identical to the
 *       fresh one. A divergence means the in-place buffer refill corrupted the replay.</li>
 *   <li><b>not degenerate</b> — the fresh generate must not collapse to a short repeating loop (the
 *       symptom of the pre-fix demotion bug), guarded by a distinct-token floor.</li>
 *   <li><b>re-prefill overwrites</b> — a different prompt in the middle must produce different output,
 *       and returning to the first prompt must reproduce its original output (proves the prefill inputs
 *       are re-written in place, not left stale — the gen-3+ failure mode the fix specifically targets).</li>
 * </ol>
 *
 * <p>Uses one isolated Qwen3.5 0.8B (Q4_K_M) model dedicated to fixed-buffer pipelines, so there is no
 * variable-buffer pipeline sharing the decoder's executor (which would introduce a cross-config
 * premature-freeze artifact unrelated to the reuse path). CUDA + model download required.</p>
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test -Dtest=TestFixedBufferDecodeReuse -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 */
@Slf4j
public class TestFixedBufferDecodeReuse {

    private static final String PROMPT =
            "Once upon a time, in a land far away, there lived a curious inventor who";
    private static final String PROMPT_B =
            "The history of scientific discovery shows that the most important breakthroughs often";

    /** Decode length per generate — long enough to expose a degenerate loop, short enough to stay fast. */
    private static final int N = 60;
    /** Number of generates on one pipeline (gen 0 fresh + gens 1..7 reused). */
    private static final int GENERATES = 8;
    /** Distinct-token floor: the pre-fix degeneration collapsed to a ~4-token loop. */
    private static final int MIN_DISTINCT = 5;

    private static SameDiff model;
    private static Tokenizer tokenizer;
    private static String modelPath;

    @BeforeAll
    public static void setup() throws Exception {
        if (System.getProperty(ND4JSystemProperties.OPTIMIZER_ENABLED) == null) {
            System.setProperty(ND4JSystemProperties.OPTIMIZER_ENABLED, "true");
        }
        String sizeLabel = System.getProperty("qwen.model.size", "0.8B");
        String quantStr = System.getProperty("qwen.quant", "Q4_K_M");

        DownloadResult dl = LLMModelDownloader.download(LLMModel.fromSizeLabel(sizeLabel), QuantType.valueOf(quantStr));
        modelPath = dl.getModelFile().getAbsolutePath();
        model = GGMLModelImport.importModel(modelPath);

        String tokenizerPath = System.getProperty("qwen.tokenizer.path");
        if (tokenizerPath != null && !tokenizerPath.isEmpty()) {
            tokenizer = HuggingFaceTokenizer.fromFile(tokenizerPath);
        } else {
            String tokenizerUrl = "https://huggingface.co/Qwen/Qwen3.5-" + sizeLabel + "/resolve/main/tokenizer.json";
            File tf = LLMModelDownloader.downloadCustom(tokenizerUrl, "qwen35-" + sizeLabel + "-tokenizer.json");
            tokenizer = HuggingFaceTokenizer.fromFile(tf.getAbsolutePath());
        }
    }

    @AfterAll
    public static void teardown() {
        model = null;
        tokenizer = null;
    }

    private static GenerationPipeline fixedBufferPipeline() throws Exception {
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(model)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(N)
                .maxPrefillLength(64)
                .maxKvCacheLength(128)
                .graphOptimizerEnabled(true)
                .dspEnabled(true)
                .build();
        return GenerationPipeline.create(cfg);
    }

    /** Variable-buffer (no fixed-buffer fast path) pipeline — the known-correct reference decoder. */
    private static GenerationPipeline variablePipeline() throws Exception {
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(model)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(N)
                .graphOptimizerEnabled(true)
                .dspEnabled(true)
                .build();
        return GenerationPipeline.create(cfg);
    }

    private static int distinct(int[] a) {
        Set<Integer> s = new HashSet<>();
        for (int x : a) s.add(x);
        return s.size();
    }

    /** Greedy pipeline over a caller-supplied model, fixed-buffer or variable-buffer. */
    private static GenerationPipeline pipelineFor(SameDiff m, boolean fixed) throws Exception {
        GenerationPipelineConfig cfg = fixed
                ? GenerationPipelineConfig.builder().decoder(m).tokenizer(tokenizer)
                    .samplingConfig(SamplingConfig.greedy()).maxNewTokens(N)
                    .maxPrefillLength(64).maxKvCacheLength(128)
                    .graphOptimizerEnabled(true).dspEnabled(true).build()
                : GenerationPipelineConfig.builder().decoder(m).tokenizer(tokenizer)
                    .samplingConfig(SamplingConfig.greedy()).maxNewTokens(N)
                    .graphOptimizerEnabled(true).dspEnabled(true).build();
        return GenerationPipeline.create(cfg);
    }

    /**
     * DIAGNOSTIC (not a gate): confirm the fixed-buffer-vs-variable divergence is REAL and not a
     * shared-executor contamination artifact — each config runs on its OWN freshly-imported model, so
     * the executors are fully isolated. If fixedFresh still != variableRef here, the fixed-buffer path
     * has a genuine accuracy gap vs the reference decoder (independent of the reuse work).
     */
    @Test
    @DisplayName("DIAG: isolated fixed-buffer vs variable-buffer (separate model imports)")
    public void diagIsolatedFixedVsVariable() throws Exception {
        SameDiff mV = GGMLModelImport.importModel(modelPath);
        int[] variableRef;
        GenerationPipeline varPipe = pipelineFor(mV, false);
        try {
            variableRef = varPipe.generate(PROMPT, N).getTokenIds();
        } finally {
            varPipe.close();
            SameDiffMemoryUtils.freeModelArrays(mV);
        }
        // Release the variable model before importing the fixed one — three resident 0.8B models
        // (shared setup model + mV + mF) exceed the 24GB physical-memory cap.
        mV = null;

        SameDiff mF = GGMLModelImport.importModel(modelPath);
        int[] fixedFresh;
        GenerationPipeline fixPipe = pipelineFor(mF, true);
        try {
            fixedFresh = fixPipe.generate(PROMPT, N).getTokenIds();
        } finally {
            fixPipe.close();
            SameDiffMemoryUtils.freeModelArrays(mF);
        }
        mF = null;

        log.info("[ISO] variableRef = {}", Arrays.toString(variableRef));
        log.info("[ISO] fixedFresh  = {}", Arrays.toString(fixedFresh));
        log.info("[ISO] fixed==variable? {}   (divergeIdx={})",
                Arrays.equals(fixedFresh, variableRef), firstDiff(fixedFresh, variableRef));
    }

    /** Index of the first differing element, or -1 if equal up to the shorter length. */
    private static int firstDiff(int[] a, int[] b) {
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) if (a[i] != b[i]) return i;
        return (a.length == b.length) ? -1 : n;
    }

    private static int extractLayerIndex(String kvInputName) {
        for (String part : kvInputName.split("\\.")) {
            try {
                return Integer.parseInt(part);
            } catch (NumberFormatException ignored) {
                // Continue to the numeric path component.
            }
        }
        return 0;
    }

    private static INDArray buildPaddedPrefillCausalMask(
            int actualLength, int paddedLength, long maxKvLength, DataType dataType) {
        int keys = (int) maxKvLength;
        float maskValue = (dataType == DataType.HALF || dataType == DataType.FLOAT16)
                ? -65504.0f : -1.0e9f;
        float[] values = new float[paddedLength * keys];
        for (int query = 0; query < paddedLength; query++) {
            int row = query * keys;
            int firstMaskedKey = query < actualLength ? query + 1 : 0;
            for (int key = firstMaskedKey; key < keys; key++) {
                values[row + key] = maskValue;
            }
        }

        INDArray mask = Nd4j.create(
                values, new long[]{1, 1, paddedLength, maxKvLength}, 'c');
        if (dataType == DataType.FLOAT) return mask;
        INDArray cast = mask.castTo(dataType);
        mask.close();
        return cast;
    }

    private static double measureLayer3AttentionSnapshotParity(
            float[] queryValues, float[] keyValues, float[] valueValues,
            int actualLength, int paddedLength, String diagnosticLabel) {
        final int batch = 1;
        final int queryHeads = 8;
        final int kvHeads = 2;
        final int headDim = 256;

        INDArray queryData = Nd4j.createFromArray(queryValues)
                .reshape(batch, paddedLength, queryHeads, headDim);
        INDArray keyData = Nd4j.createFromArray(keyValues)
                .reshape(batch, paddedLength, kvHeads, headDim);
        INDArray valueData = Nd4j.createFromArray(valueValues)
                .reshape(batch, paddedLength, kvHeads, headDim);
        INDArray biasData = buildPaddedPrefillCausalMask(
                actualLength, paddedLength, paddedLength, DataType.FLOAT);

        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("query", queryData);
        placeholders.put("value", valueData);
        placeholders.put("key", keyData);
        placeholders.put("attention_bias", biasData);

        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        INDArray reference = null;
        double overallMaxAbsDiff = 0.0;
        try {
            try (SameDiff nativeGraph = SameDiff.create()) {
                SDVariable query = nativeGraph.placeHolder(
                        "query", DataType.FLOAT, batch, paddedLength, queryHeads, headDim);
                SDVariable value = nativeGraph.placeHolder(
                        "value", DataType.FLOAT, batch, paddedLength, kvHeads, headDim);
                SDVariable key = nativeGraph.placeHolder(
                        "key", DataType.FLOAT, batch, paddedLength, kvHeads, headDim);
                SDVariable bias = nativeGraph.placeHolder(
                        "attention_bias", DataType.FLOAT, batch, 1, paddedLength, paddedLength);
                SDVariable emptyKeyCache = nativeGraph.constant(Nd4j.empty(DataType.FLOAT));
                SDVariable emptyValueCache = nativeGraph.constant(Nd4j.empty(DataType.FLOAT));
                SDVariable emptyCachePosition = nativeGraph.constant(Nd4j.empty(DataType.INT64));
                SDVariable attention = new DotProductAttentionV2(
                        nativeGraph, query, value, key, null, null,
                        emptyKeyCache, emptyValueCache, emptyCachePosition, bias,
                        0.0, 0.0, false, false).outputVariable();
                nativeGraph.updateVariableNameAndReference(attention, "attention");
                nativeGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
                reference = nativeGraph.output(placeholders, "attention").get("attention").dup();
            }

            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("ATTENTION");
            float[] expected = reference.toFloatVector();
            try (SameDiff tritonGraph = SameDiff.create()) {
                SDVariable query = tritonGraph.placeHolder(
                        "query", DataType.FLOAT, batch, paddedLength, queryHeads, headDim);
                SDVariable value = tritonGraph.placeHolder(
                        "value", DataType.FLOAT, batch, paddedLength, kvHeads, headDim);
                SDVariable key = tritonGraph.placeHolder(
                        "key", DataType.FLOAT, batch, paddedLength, kvHeads, headDim);
                SDVariable bias = tritonGraph.placeHolder(
                        "attention_bias", DataType.FLOAT, batch, 1, paddedLength, paddedLength);
                SDVariable emptyKeyCache = tritonGraph.constant(Nd4j.empty(DataType.FLOAT));
                SDVariable emptyValueCache = tritonGraph.constant(Nd4j.empty(DataType.FLOAT));
                SDVariable emptyCachePosition = tritonGraph.constant(Nd4j.empty(DataType.INT64));
                SDVariable attention = new DotProductAttentionV2(
                        tritonGraph, query, value, key, null, null,
                        emptyKeyCache, emptyValueCache, emptyCachePosition, bias,
                        0.0, 0.0, false, false).outputVariable();
                tritonGraph.updateVariableNameAndReference(attention, "attention");
                tritonGraph.setGraphExecutionMode(GraphExecutionMode.TRITON);

                for (int step = 0; step < 4; step++) {
                    float[] actual = tritonGraph.output(placeholders, "attention")
                            .get("attention").toFloatVector();
                    long mismatchCount = 0;
                    double maxAbsDiff = 0.0;
                    int maxDiffIndex = -1;
                    for (int i = 0; i < expected.length; i++) {
                        double absDiff = Math.abs((double) expected[i] - actual[i]);
                        if (absDiff > maxAbsDiff) {
                            maxAbsDiff = absDiff;
                            maxDiffIndex = i;
                        }
                        if (absDiff > 1.0e-6) mismatchCount++;
                    }
                    overallMaxAbsDiff = Math.max(overallMaxAbsDiff, maxAbsDiff);
                    log.info("{} step={} mismatches={}/{} "
                                    + "maxAbsDiff={} maxDiffIndex={} native={} triton={}",
                            diagnosticLabel, step, mismatchCount, expected.length,
                            maxAbsDiff, maxDiffIndex,
                            maxDiffIndex < 0 ? 0.0f : expected[maxDiffIndex],
                            maxDiffIndex < 0 ? 0.0f : actual[maxDiffIndex]);
                }

                DspPlanAssertions.assertOpCompiled(
                        tritonGraph, "dot_product_attention_v2",
                        diagnosticLabel);
                DspPlanAssertions.assertAllSegmentsCompiledWith(
                        tritonGraph, "Triton GPU",
                        diagnosticLabel);
            }
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
            if (reference != null && !reference.wasClosed()) reference.close();
            queryData.close();
            keyData.close();
            valueData.close();
            biasData.close();
        }
        return overallMaxAbsDiff;
    }

    /** Fixed-buffer greedy pipeline with explicit padding / KV-cache sizes. */
    private static GenerationPipeline pipelineForPad(SameDiff m, int maxPrefill, int maxKv) throws Exception {
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(m).tokenizer(tokenizer).samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(N).maxPrefillLength(maxPrefill).maxKvCacheLength(maxKv)
                .graphOptimizerEnabled(true).dspEnabled(true).build();
        return GenerationPipeline.create(cfg);
    }

    /**
     * DISCRIMINATOR (numerical vs structural): the prompt is ~15 tokens. Padding it to 64 (KV=128) vs
     * 96 (KV=192) MUST produce identical greedy output — the padding positions are masked, so the real
     * tokens attend only to 0..14 either way and the decode starts at the same real position. The ONLY
     * difference is buffer/plan SHAPE, which changes DSP kernel/tile/TF32 selection but not the math.
     * <ul>
     *   <li>pad64 != pad96  ⇒ the fixed-buffer output depends on a logically-irrelevant shape ⇒ the
     *       divergence is NUMERICAL (kernel selection tips a near-tie), not a structural bug.</li>
     *   <li>pad64 == pad96  ⇒ the fixed-buffer path is shape-robust ⇒ the fixed-vs-variable gap comes
     *       from something structural (dig further: mask/position/KV handling).</li>
     * </ul>
     * Isolated model imports (free between) to avoid shared-executor contamination and the 24GB cap.
     */
    @Test
    @DisplayName("DIAG: fixed-buffer output invariance to padding length (numerical vs structural)")
    public void diagPaddingSensitivity() throws Exception {
        SameDiff m1 = GGMLModelImport.importModel(modelPath);
        int[] pad64;
        GenerationPipeline p1 = pipelineForPad(m1, 64, 128);
        try {
            pad64 = p1.generate(PROMPT, N).getTokenIds();
        } finally {
            p1.close();
        }
        SameDiffMemoryUtils.freeModelArrays(m1);
        m1 = null;

        SameDiff m2 = GGMLModelImport.importModel(modelPath);
        int[] pad96;
        GenerationPipeline p2 = pipelineForPad(m2, 96, 192);
        try {
            pad96 = p2.generate(PROMPT, N).getTokenIds();
        } finally {
            p2.close();
        }
        SameDiffMemoryUtils.freeModelArrays(m2);
        m2 = null;

        log.info("[PAD] pad64/kv128 = {}", Arrays.toString(pad64));
        log.info("[PAD] pad96/kv192 = {}", Arrays.toString(pad96));
        log.info("[PAD] equal? {}   (divergeIdx={}) -> {}",
                Arrays.equals(pad64, pad96), firstDiff(pad64, pad96),
                Arrays.equals(pad64, pad96) ? "shape-robust (structural gap elsewhere)"
                                            : "shape-sensitive => NUMERICAL near-tie");
    }

    /**
     * DIAGNOSTIC (not a gate): establish ground truth for the reuse divergence by comparing fixed-buffer
     * fresh (gen 0) and reused (gen 1) against the variable-buffer reference decoder. Runs the fixed
     * pipeline FIRST (clean, fresh executor) then the variable reference (its path resets the executor),
     * so both references are uncontaminated. Logs the three token arrays and the equality matrix.
     */
    @Test
    @DisplayName("DIAG: fixed fresh/reuse vs variable-buffer reference")
    public void diagReferenceComparison() throws Exception {
        int[] fixedFresh;
        int[] fixedReuse;
        GenerationPipeline fixedPipe = fixedBufferPipeline();
        try {
            fixedFresh = fixedPipe.generate(PROMPT, N).getTokenIds();
            fixedReuse = fixedPipe.generate(PROMPT, N).getTokenIds();
        } finally {
            fixedPipe.close();
        }

        int[] variableRef;
        GenerationPipeline varPipe = variablePipeline();
        try {
            variableRef = varPipe.generate(PROMPT, N).getTokenIds();
        } finally {
            varPipe.close();
        }

        log.info("[DIAG] variableRef = {}", Arrays.toString(variableRef));
        log.info("[DIAG] fixedFresh  = {}", Arrays.toString(fixedFresh));
        log.info("[DIAG] fixedReuse  = {}", Arrays.toString(fixedReuse));
        log.info("[DIAG] fresh==ref? {}   reuse==ref? {}   fresh==reuse? {}",
                Arrays.equals(fixedFresh, variableRef),
                Arrays.equals(fixedReuse, variableRef),
                Arrays.equals(fixedFresh, fixedReuse));
    }

    /**
     * Core gate: gen 0 (fresh, no cached state) then gens 1..7 (each reusing the cached frozen state)
     * must be token-identical under greedy decoding on the same prompt, and gen 0 must not be degenerate.
     */
    @Test
    @DisplayName("fixed-buffer reuse across generates is consistent and coherent (greedy)")
    public void reuseAcrossGeneratesIsConsistentAndCoherent() throws Exception {
        GenerationPipeline pipe = fixedBufferPipeline();
        try {
            int[] gen0 = pipe.generate(PROMPT, N).getTokenIds();      // fresh — builds + freezes the plan
            assertTrue(gen0.length >= 8,
                    "gen 0 produced too few tokens (" + gen0.length + "): " + Arrays.toString(gen0));
            assertTrue(distinct(gen0) >= MIN_DISTINCT,
                    "gen 0 looks degenerate (only " + distinct(gen0) + " distinct tokens): " + Arrays.toString(gen0));

            for (int g = 1; g < GENERATES; g++) {
                int[] genN = pipe.generate(PROMPT, N).getTokenIds();  // reuse — replays the frozen plan
                assertArrayEquals(gen0, genN,
                        "reuse gen " + g + " diverged from fresh gen 0 — in-place buffer refill corrupted "
                        + "decode.\n  fresh=" + Arrays.toString(gen0) + "\n  gen" + g + "=" + Arrays.toString(genN));
            }
            log.info("[fixed-buffer-reuse] {} generates all matched, {} distinct tokens: {}",
                    GENERATES, distinct(gen0), Arrays.toString(gen0));
        } finally {
            closePipelineWithBufferDiagnostics(pipe);
        }
    }

    /**
     * Lifecycle gate: after one full create/capture/close cycle has established process-wide CUDA
     * caches, a second equivalent cycle must not retain another model-sized pool allocation.
     */
    @Test
    @DisplayName("sequential fixed-buffer pipeline close does not retain a graph-pinned model copy")
    public void sequentialPipelineCloseDoesNotRetainGraphPinnedModelCopy() throws Exception {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int device = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        long[] baseline = memoryPoolStats(nativeOps, device);
        long[][] afterClose = new long[2][];

        for (int cycle = 0; cycle < afterClose.length; cycle++) {
            GenerationPipeline pipe = fixedBufferPipeline();
            try {
                int[] tokens = pipe.generate(PROMPT, N).getTokenIds();
                assertTrue(tokens.length >= 8,
                        "cycle " + cycle + " produced too few tokens: " + Arrays.toString(tokens));
            } finally {
                closePipelineWithBufferDiagnostics(pipe);
            }
            afterClose[cycle] = memoryPoolStats(nativeOps, device);
        }

        long retainedGrowth = Math.max(0L, afterClose[1][0] - afterClose[0][0]);
        long maxExpectedGrowth = 1024L * 1024L * 1024L;
        log.info("[fixed-buffer-lifecycle] device={} baselineUsed={}MB afterFirst={}MB "
                        + "afterSecond={}MB retainedGrowth={}MB",
                device,
                baseline[0] / (1024 * 1024),
                afterClose[0][0] / (1024 * 1024),
                afterClose[1][0] / (1024 * 1024),
                retainedGrowth / (1024 * 1024));
        assertTrue(retainedGrowth <= maxExpectedGrowth,
                "Second fixed-buffer pipeline close retained "
                        + retainedGrowth / (1024 * 1024)
                        + "MB beyond the warmed first-cycle baseline; expected at most "
                        + maxExpectedGrowth / (1024 * 1024)
                        + "MB. This is consistent with a graph-pinned model allocation leak.");
    }

    /**
     * Regression for sharded SameDiff load ownership. A load should allocate one model copy on the
     * active device, and closing that graph must release it without retaining a delayed device copy.
     */
    @Test
    @DisplayName("sharded SameDiff round trip releases temporary model copies")
    public void shardedSameDiffRoundTripReleasesTemporaryModelCopies() throws Exception {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        File tempDir = File.createTempFile("samediff-roundtrip-", "");
        assertTrue(tempDir.delete() && tempDir.mkdirs(),
                "Could not create serializer discriminator directory: " + tempDir);
        File baseFile = new File(tempDir, "model.sdnb");

        SameDiff loaded = null;
        long[][] baseline = allMemoryPoolStats(nativeOps, numDevices);
        long[][] afterSave;
        long[][] afterLoad;
        long[][] afterClose;
        Map<Integer, long[]> loadedBuffersByDevice = Collections.emptyMap();
        try {
            SameDiffSerializer.save(model, baseFile, true, Collections.emptyMap());
            afterSave = allMemoryPoolStats(nativeOps, numDevices);

            loaded = SameDiffSerializer.load(baseFile, true);
            assertNotNull(loaded, "Sharded SameDiff load returned null");
            loadedBuffersByDevice = loadedBufferBytesByDevice(loaded);
            afterLoad = allMemoryPoolStats(nativeOps, numDevices);

            loaded.close();
            SameDiffMemoryUtils.freeModelArrays(loaded);
            loaded = null;
            afterClose = allMemoryPoolStats(nativeOps, numDevices);
        } finally {
            if (loaded != null) {
                loaded.close();
                SameDiffMemoryUtils.freeModelArrays(loaded);
            }
            deleteRecursively(tempDir);
        }

        for (int device = 0; device < numDevices; device++) {
            log.info("[samediff-roundtrip-lifecycle] currentDevice={} device={} baseline={}MB "
                            + "afterSave={}MB afterLoad={}MB afterClose={}MB saveDelta={}MB "
                            + "loadDelta={}MB retained={}MB reserved={}MB->{}MB",
                    currentDevice, device,
                    baseline[device][0] / (1024 * 1024),
                    afterSave[device][0] / (1024 * 1024),
                    afterLoad[device][0] / (1024 * 1024),
                    afterClose[device][0] / (1024 * 1024),
                    (afterSave[device][0] - baseline[device][0]) / (1024 * 1024),
                    (afterLoad[device][0] - afterSave[device][0]) / (1024 * 1024),
                    (afterClose[device][0] - baseline[device][0]) / (1024 * 1024),
                    baseline[device][1] / (1024 * 1024),
                    afterClose[device][1] / (1024 * 1024));
        }
        for (Map.Entry<Integer, long[]> entry : loadedBuffersByDevice.entrySet()) {
            log.info("[samediff-roundtrip-loaded-buffers] targetDevice={} buffers={} bytes={}MB",
                    entry.getKey(), entry.getValue()[0], entry.getValue()[1] / (1024 * 1024));
        }

        long maxLifecycleOverheadBytes = 512L * 1024L * 1024L;
        long[] currentDeviceBuffers = loadedBuffersByDevice.get(currentDevice);
        assertNotNull(currentDeviceBuffers,
                "Loaded graph has no buffers associated with current device " + currentDevice);

        long currentLoadDelta = Math.max(0L,
                afterLoad[currentDevice][0] - afterSave[currentDevice][0]);
        assertTrue(currentLoadDelta <= currentDeviceBuffers[1] + maxLifecycleOverheadBytes,
                "Sharded load allocated "
                        + currentLoadDelta / (1024 * 1024)
                        + "MB for "
                        + currentDeviceBuffers[1] / (1024 * 1024)
                        + "MB of owned model buffers. This indicates an unowned device copy.");

        for (int device = 0; device < numDevices; device++) {
            long retainedBytes = Math.max(0L, afterClose[device][0] - baseline[device][0]);
            assertTrue(retainedBytes <= maxLifecycleOverheadBytes,
                    "Sharded SameDiff close retained "
                            + retainedBytes / (1024 * 1024)
                            + "MB on device "
                            + device
                            + "; expected at most "
                            + maxLifecycleOverheadBytes / (1024 * 1024)
                            + "MB of allocator noise.");
        }
    }

    private static void deleteRecursively(File file) {
        if (file == null || !file.exists()) {
            return;
        }
        File[] children = file.listFiles();
        if (children != null) {
            for (File child : children) {
                deleteRecursively(child);
            }
        }
        assertTrue(file.delete(), "Could not delete serializer discriminator file: " + file);
    }

    private static Map<Integer, long[]> loadedBufferBytesByDevice(SameDiff graph) {
        Map<Integer, long[]> byDevice = new LinkedHashMap<>();
        Set<DataBuffer> seenBuffers = Collections.newSetFromMap(new IdentityHashMap<>());
        for (SDVariable variable : graph.variables()) {
            INDArray array = graph.getArrForVarName(variable.name());
            if (array == null || array.data() == null) {
                continue;
            }
            DataBuffer buffer = array.data();
            if (!seenBuffers.add(buffer)) {
                continue;
            }
            long[] summary = byDevice.computeIfAbsent(buffer.targetDevice(), ignored -> new long[2]);
            summary[0]++;
            summary[1] += buffer.length() * (long) buffer.getElementSize();
        }
        return byDevice;
    }

    private static long[][] allMemoryPoolStats(NativeOps nativeOps, int numDevices) {
        long[][] result = new long[numDevices][];
        for (int device = 0; device < numDevices; device++) {
            result[device] = memoryPoolStats(nativeOps, device);
        }
        return result;
    }

    private static void closePipelineWithBufferDiagnostics(GenerationPipeline pipe) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int device = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        long[] poolBefore = memoryPoolStats(nativeOps, device);
        nativeOps.dbCloseResetDiagnostics();

        pipe.close();

        long[] poolAfter = memoryPoolStats(nativeOps, device);
        try (LongPointer nativeStats = new LongPointer(9)) {
            nativeOps.dbCloseGetDiagnostics(nativeStats);
            log.info("[fixed-buffer-close] device={} poolUsed={}MB->{}MB poolReserved={}MB->{}MB "
                            + "dbClose total={} null={} constant={} alreadyClosed={} noDataBuffer={} "
                            + "notOwner={} deviceError={} deleted={} freedBytes={}MB",
                    device,
                    poolBefore[0] / (1024 * 1024), poolAfter[0] / (1024 * 1024),
                    poolBefore[1] / (1024 * 1024), poolAfter[1] / (1024 * 1024),
                    nativeStats.get(0), nativeStats.get(1), nativeStats.get(2), nativeStats.get(3),
                    nativeStats.get(4), nativeStats.get(5), nativeStats.get(6), nativeStats.get(7),
                    nativeStats.get(8) / (1024 * 1024));
        } finally {
            nativeOps.dbCloseResetDiagnostics();
        }
    }

    private static long[] memoryPoolStats(NativeOps nativeOps, int device) {
        try (LongPointer poolStats = new LongPointer(2);
             LongPointer reservedStats = new LongPointer(1)) {
            nativeOps.getMemoryPoolStats(device, poolStats, reservedStats);
            return new long[]{poolStats.get(0), reservedStats.get(0)};
        }
    }

    /**
     * Execute the exact padded prefill graph repeatedly and compare every output the
     * pipeline already requests. This keeps the plan shape and requested-output set
     * unchanged while locating the first layer boundary that diverges when Triton
     * sections become available.
     */
    @Test
    @DisplayName("padded prefill requested outputs remain exact at the Triton compile transition")
    public void paddedPrefillOutputsRemainExactAtCompileTransition() throws Exception {
        final int prefillLength = 64;
        final int maxKvLength = 128;
        GenerationPipeline pipe = fixedBufferPipeline();
        Map<String, INDArray> prefillInputs = new HashMap<>();
        Set<INDArray> ownedInputs = Collections.newSetFromMap(new IdentityHashMap<>());

        try {
            int[] promptTokenIds = tokenizer.encodePrompt(PROMPT, null).getIds();
            int actualPrefillLength = promptTokenIds.length;
            assertTrue(actualPrefillLength > 0 && actualPrefillLength <= prefillLength,
                    "Prompt length must fit the fixed prefill buffer");

            int[] effectiveTokenIds = new int[prefillLength];
            System.arraycopy(promptTokenIds, 0, effectiveTokenIds, 0, actualPrefillLength);

            ModelIOConfig ioConfig = pipe.getIoConfig();
            String inputIdsName = ioConfig.getInputIdsName() != null
                    ? ioConfig.getInputIdsName() : "input_ids";
            String logitsName = ioConfig.getLogitsOutputName() != null
                    ? ioConfig.getLogitsOutputName() : "lm_logits";
            String positionOffsetName = ioConfig.getPositionOffsetName();
            String cachePositionName = ioConfig.getCachePositionName();
            String causalMaskName = ioConfig.getCausalMaskName();

            prefillInputs.put(inputIdsName, Nd4j.createFromArray(effectiveTokenIds)
                    .reshape(1, prefillLength).castTo(DataType.INT64));
            if (positionOffsetName != null && model.hasVariable(positionOffsetName)) {
                prefillInputs.put(positionOffsetName, Nd4j.scalar(DataType.INT64, 0));
            }
            if (cachePositionName != null && model.hasVariable(cachePositionName)) {
                prefillInputs.put(cachePositionName, Nd4j.scalar(DataType.INT64, 0));
            }
            if (model.hasVariable("actual_sequence_length")) {
                prefillInputs.put("actual_sequence_length",
                        Nd4j.scalar(DataType.INT64, actualPrefillLength));
            }
            if (causalMaskName != null && model.hasVariable(causalMaskName)) {
                DataType maskType = model.getVariable(causalMaskName).dataType();
                prefillInputs.put(causalMaskName,
                        buildPaddedPrefillCausalMask(
                                actualPrefillLength, prefillLength, maxKvLength, maskType));
            }

            ModelIOConfig.KVCacheNames kvInputNames =
                    ModelIOConfig.findKVCacheInputNames(model);
            assertNotNull(kvInputNames, "Qwen GGUF model must expose in-graph KV inputs");
            for (String keyName : kvInputNames.keyNames) {
                if (model.hasVariable(keyName)) {
                    prefillInputs.put(keyName,
                            Nd4j.empty(model.getVariable(keyName).dataType()));
                }
            }
            for (String valueName : kvInputNames.valueNames) {
                if (model.hasVariable(valueName)) {
                    prefillInputs.put(valueName,
                            Nd4j.empty(model.getVariable(valueName).dataType()));
                }
            }

            List<ModelIOConfig.RecurrentStatePair> recurrentStates =
                    ModelIOConfig.findRecurrentStatePairs(model, ioConfig);
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                if (!model.hasVariable(pair.inputName)) continue;
                long[] stateShape =
                        GenerationPipeline.deriveRecurrentStateShape(model, pair.inputName);
                assertNotNull(stateShape,
                        "Could not derive recurrent prefill state shape for " + pair.inputName);
                prefillInputs.put(pair.inputName,
                        Nd4j.zeros(model.getVariable(pair.inputName).dataType(), stateShape));
            }

            List<String> outputNames = new ArrayList<>();
            outputNames.add(logitsName);
            for (String keyName : kvInputNames.keyNames) {
                int layer = extractLayerIndex(keyName);
                outputNames.add("k_rope_" + layer);
                outputNames.add("v_heads_" + layer);
            }
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                outputNames.add(pair.outputName);
            }

            // Pin exact operation boundaries, not merely downstream KV/state symptoms.
            // The recurrent checkpoints cover standalone swish and its consumers; the first
            // full-attention checkpoints cover Q/K projection, per-head RMSNorm, and RoPE.
            List<String> compileTransitionCheckpoints = Arrays.asList(
                    "model.layers.0.input_layernorm",
                    "gdn_qkv_0",
                    "gdn_conv_0",
                    "gdn_gate_proj_0",
                    "gdn_z_reshaped_0",
                    "gdn_gate_act_0",
                    "gdn_out_0",
                    "model.layers.0.gdn.ssm_norm_0",
                    "gdn_gated_0",
                    "gdn_proj_0",
                    "post_attn_0",
                    "model.layers.0.post_attention_layernorm",
                    "gate_0",
                    "up_0",
                    "swish_1",
                    "swiglu_0",
                    "down_0",
                    "layer_out_0",
                    "model.layers.1.input_layernorm",
                    "gdn_qkv_1",
                    "gdn_conv_1",
                    "model.layers.3.input_layernorm",
                    "q_full_3",
                    "k_3",
                    "v_3",
                    "qg_reshaped_3",
                    "q_3",
                    "attn_gate_3",
                    "k_heads_3",
                    "model.layers.3.self_attn.q_norm_3",
                    "model.layers.3.self_attn.k_norm_3",
                    "q_rope_3",
                    "attn_out_3",
                    "attn_flat_3",
                    "gate_sigmoid_3",
                    "gated_attn_3",
                    "attn_proj_3",
                    "post_attn_3",
                    "model.layers.3.post_attention_layernorm",
                    "gate_3",
                    "up_3",
                    "swiglu_3",
                    "down_3",
                    "layer_out_3",
                    "model.layers.4.input_layernorm",
                    "gdn_qkv_4",
                    "gdn_conv_4");
            for (String checkpoint : compileTransitionCheckpoints) {
                assertTrue(model.hasVariable(checkpoint),
                        "Qwen graph is missing compile-transition checkpoint " + checkpoint);
                outputNames.add(checkpoint);
            }

            ownedInputs.addAll(prefillInputs.values());

            Map<String, float[]> baseline = new LinkedHashMap<>();
            StringBuilder differences = new StringBuilder();
            int mismatchingOutputs = 0;
            boolean sawTriton = false;

            for (int step = 0; step < 4; step++) {
                Map<String, INDArray> outputs = model.output(
                        prefillInputs, outputNames.toArray(new String[0]));

                int phase = DspPlanAssertions.getPlanPhase(model);
                int segmentCount = model.dsp().numSegments();
                StringBuilder backends = new StringBuilder();
                for (int segment = 0; segment < segmentCount; segment++) {
                    String backend = DspPlanAssertions.getSegmentCompiledBackend(model, segment);
                    if (segment > 0) backends.append(',');
                    backends.append(segment).append('=').append(backend);
                    if ("Triton GPU".equals(backend)) sawTriton = true;
                }

                int sampledToken = -1;
                int stepMismatchOutputs = 0;
                for (String outputName : outputNames) {
                    INDArray output = outputs.get(outputName);
                    assertNotNull(output, "Missing prefill output " + outputName + " at step " + step);

                    INDArray comparable = output;
                    if (outputName.equals(ioConfig.getLogitsOutputName()) && output.rank() == 3) {
                        comparable = output.get(
                                NDArrayIndex.point(0),
                                NDArrayIndex.point(actualPrefillLength - 1),
                                NDArrayIndex.all());
                    }
                    float[] values = comparable.toFloatVector();

                    if (step == 0 && (outputName.equals("q_rope_3")
                            || outputName.equals("k_rope_3")
                            || outputName.equals("v_heads_3")
                            || outputName.equals("attn_out_3"))) {
                        float minValue = Float.POSITIVE_INFINITY;
                        float maxValue = Float.NEGATIVE_INFINITY;
                        double maxAbsValue = 0.0;
                        double l1Value = 0.0;
                        for (float value : values) {
                            minValue = Math.min(minValue, value);
                            maxValue = Math.max(maxValue, value);
                            maxAbsValue = Math.max(maxAbsValue, Math.abs((double) value));
                            l1Value += Math.abs((double) value);
                        }
                        log.info("PREFILL_LAYOUT output={} shape={} stride={} order={} ews={} view={} "
                                        + "min={} max={} maxAbs={} l1={}",
                                outputName, Arrays.toString(output.shape()),
                                Arrays.toString(output.stride()), output.ordering(),
                                output.elementWiseStride(), output.isView(),
                                minValue, maxValue, maxAbsValue, l1Value);
                    }

                    if (outputName.equals(ioConfig.getLogitsOutputName())) {
                        float best = Float.NEGATIVE_INFINITY;
                        for (int i = 0; i < values.length; i++) {
                            if (values[i] > best) {
                                best = values[i];
                                sampledToken = i;
                            }
                        }
                    }

                    if (step == 0) {
                        baseline.put(outputName, values);
                        continue;
                    }

                    float[] expected = baseline.get(outputName);
                    assertNotNull(expected, "No baseline captured for " + outputName);
                    assertEquals(expected.length, values.length,
                            "Output length changed for " + outputName + " at step " + step);

                    long mismatchCount = 0;
                    double maxAbsDiff = 0.0;
                    int firstMismatch = -1;
                    for (int i = 0; i < expected.length; i++) {
                        if (Float.floatToRawIntBits(expected[i])
                                != Float.floatToRawIntBits(values[i])) {
                            mismatchCount++;
                            if (firstMismatch < 0) firstMismatch = i;
                        }
                        maxAbsDiff = Math.max(
                                maxAbsDiff, Math.abs((double) expected[i] - values[i]));
                    }

                    if (mismatchCount > 0) {
                        stepMismatchOutputs++;
                        mismatchingOutputs++;
                        differences.append("\nstep=").append(step)
                                .append(" output=").append(outputName)
                                .append(" mismatches=").append(mismatchCount)
                                .append('/').append(expected.length)
                                .append(" maxAbsDiff=").append(maxAbsDiff)
                                .append(" firstIndex=").append(firstMismatch)
                                .append(" native=").append(expected[firstMismatch])
                                .append(" compiled=").append(values[firstMismatch]);
                        log.info("PREFILL_OUTPUT_DIFF step={} output={} mismatches={}/{} "
                                        + "maxAbsDiff={} firstIndex={} native={} compiled={}",
                                step, outputName, mismatchCount, expected.length,
                                maxAbsDiff, firstMismatch,
                                expected[firstMismatch], values[firstMismatch]);
                    }
                }

                log.info("PREFILL_TRANSITION step={} phase={} backends=[{}] token={} "
                                + "mismatchingOutputs={}",
                        step, phase, backends, sampledToken, stepMismatchOutputs);
            }

            float[] querySnapshot = baseline.get("q_rope_3");
            float[] keySnapshot = baseline.get("k_rope_3");
            float[] valueSnapshot = baseline.get("v_heads_3");
            assertNotNull(querySnapshot, "Missing native layer-3 Q snapshot");
            assertNotNull(keySnapshot, "Missing native layer-3 K snapshot");
            assertNotNull(valueSnapshot, "Missing native layer-3 V snapshot");
            double snapshotAttentionMaxAbsDiff = measureLayer3AttentionSnapshotParity(
                    querySnapshot, keySnapshot, valueSnapshot,
                    actualPrefillLength, prefillLength,
                    "PREFILL_SNAPSHOT_DPA_DENSE_VALUE_PARITY");

            float[] oneHotValueSnapshot = new float[valueSnapshot.length];
            for (int sequence = 0; sequence < prefillLength; sequence++) {
                for (int head = 0; head < 2; head++) {
                    for (int dimension = 0; dimension < 256; dimension++) {
                        if (dimension % prefillLength == sequence) {
                            int index = (sequence * 2 + head) * 256 + dimension;
                            oneHotValueSnapshot[index] = 1.0f;
                        }
                    }
                }
            }
            double snapshotProbabilityMaxAbsDiff = measureLayer3AttentionSnapshotParity(
                    querySnapshot, keySnapshot, oneHotValueSnapshot,
                    actualPrefillLength, prefillLength,
                    "PREFILL_SNAPSHOT_DPA_PROBABILITY_PARITY");

            float[] oneHotKeySnapshot = new float[keySnapshot.length];
            for (int sequence = 0; sequence < prefillLength; sequence++) {
                for (int head = 0; head < 2; head++) {
                    int dimension = (sequence * 17 + head * 31) % 256;
                    int index = (sequence * 2 + head) * 256 + dimension;
                    oneHotKeySnapshot[index] = 16.0f;
                }
            }
            double singleProductProbabilityMaxAbsDiff =
                    measureLayer3AttentionSnapshotParity(
                            querySnapshot, oneHotKeySnapshot, oneHotValueSnapshot,
                            actualPrefillLength, prefillLength,
                            "PREFILL_SNAPSHOT_DPA_SINGLE_PRODUCT_PROBABILITY_PARITY");
            int singleTileLength = actualPrefillLength;
            float[] singleTileQuerySnapshot = Arrays.copyOf(
                    querySnapshot, singleTileLength * 8 * 256);
            float[] singleTileKeySnapshot = Arrays.copyOf(
                    oneHotKeySnapshot, singleTileLength * 2 * 256);
            float[] singleTileValueSnapshot =
                    new float[singleTileLength * 2 * 256];
            for (int sequence = 0; sequence < singleTileLength; sequence++) {
                for (int head = 0; head < 2; head++) {
                    for (int dimension = 0; dimension < 256; dimension++) {
                        if (dimension % singleTileLength == sequence) {
                            int index = (sequence * 2 + head) * 256 + dimension;
                            singleTileValueSnapshot[index] = 1.0f;
                        }
                    }
                }
            }
            double singleTileProbabilityMaxAbsDiff =
                    measureLayer3AttentionSnapshotParity(
                            singleTileQuerySnapshot, singleTileKeySnapshot,
                            singleTileValueSnapshot, singleTileLength, singleTileLength,
                            "PREFILL_SNAPSHOT_DPA_SINGLE_TILE_PROBABILITY_PARITY");
            int twoKeyLength = 2;
            float[] twoKeyQuerySnapshot = Arrays.copyOf(
                    querySnapshot, twoKeyLength * 8 * 256);
            float[] twoKeySnapshot = Arrays.copyOf(
                    oneHotKeySnapshot, twoKeyLength * 2 * 256);
            float[] twoKeyValueSnapshot = new float[twoKeyLength * 2 * 256];
            for (int sequence = 0; sequence < twoKeyLength; sequence++) {
                for (int head = 0; head < 2; head++) {
                    for (int dimension = 0; dimension < 256; dimension++) {
                        if (dimension % twoKeyLength == sequence) {
                            int index = (sequence * 2 + head) * 256 + dimension;
                            twoKeyValueSnapshot[index] = 1.0f;
                        }
                    }
                }
            }
            double twoKeyProbabilityMaxAbsDiff =
                    measureLayer3AttentionSnapshotParity(
                            twoKeyQuerySnapshot, twoKeySnapshot, twoKeyValueSnapshot,
                            twoKeyLength, twoKeyLength,
                            "PREFILL_SNAPSHOT_DPA_TWO_KEY_PROBABILITY_PARITY");
            log.info("PREFILL_SNAPSHOT_DPA_SUMMARY denseValueMaxAbsDiff={} "
                            + "probabilityMaxAbsDiff={} singleProductProbabilityMaxAbsDiff={} "
                            + "singleTileProbabilityMaxAbsDiff={} twoKeyProbabilityMaxAbsDiff={}",
                    snapshotAttentionMaxAbsDiff, snapshotProbabilityMaxAbsDiff,
                    singleProductProbabilityMaxAbsDiff,
                    singleTileProbabilityMaxAbsDiff, twoKeyProbabilityMaxAbsDiff);

            assertTrue(sawTriton,
                    "The prefill lifecycle never compiled a Triton segment; discriminator was not exercised");
            assertEquals(0, mismatchingOutputs,
                    "Padded prefill outputs changed across the compile transition:" + differences);
        } finally {
            pipe.close();
            for (INDArray input : ownedInputs) {
                if (input != null && !input.wasClosed()) input.close();
            }
        }
    }

    /**
     * First-token compile-transition gate. Token 0 is sampled directly from prefill logits before
     * the pipeline's internal decode warmup. Requesting the two-token prefill+warmup prefix leaves
     * no native-loop budget, while the third call still advances the reused prefill plan to its
     * first compiled Triton execution.
     */
    @Test
    @DisplayName("fixed-buffer first prefill token remains deterministic at the Triton compile transition")
    public void firstPrefillTokenIsConsistentAtCompileTransition() throws Exception {
        GenerationPipeline pipe = fixedBufferPipeline();
        try {
            int[] expected = pipe.generate(PROMPT, 2).getTokenIds();
            assertEquals(2, expected.length, "prefill+warmup reference must honor the two-token budget");
            int expectedFirstToken = expected[0];

            for (int generation = 1; generation < 4; generation++) {
                int[] actual = pipe.generate(PROMPT, 2).getTokenIds();
                assertEquals(2, actual.length,
                        "prefill+warmup generation " + generation + " must honor the two-token budget");
                assertEquals(expectedFirstToken, actual[0],
                        "prefill token for generation " + generation
                                + " diverged at the reused-plan compile transition");
            }
        } finally {
            pipe.close();
        }
    }

    /**
     * Re-prefill overwrite: interleave a different prompt. B must differ from A, and A after B must
     * reproduce A — proving the prefill input tensors are re-written in place (not left stale), which
     * is the gen-3+ silent-degeneration failure mode the reuse fix specifically has to handle.
     */
    @Test
    @DisplayName("fixed-buffer reuse re-prefills correctly on a prompt change (A, B, A)")
    public void reusePreservesCorrectnessAcrossPromptChange() throws Exception {
        GenerationPipeline pipe = fixedBufferPipeline();
        try {
            int[] a1 = pipe.generate(PROMPT, N).getTokenIds();       // fresh
            int[] b  = pipe.generate(PROMPT_B, N).getTokenIds();      // reuse + re-prefill B
            int[] a2 = pipe.generate(PROMPT, N).getTokenIds();       // reuse + re-prefill A again

            assertArrayEquals(a1, a2,
                    "re-prefill of prompt A after B diverged — prefill-input overwrite is stale.\n"
                    + "  A(first)=" + Arrays.toString(a1) + "\n  A(after B)=" + Arrays.toString(a2));
            assertFalse(Arrays.equals(a1, b),
                    "different prompts produced identical output — re-prefill is not overwriting the KV.\n"
                    + "  A=" + Arrays.toString(a1) + "\n  B=" + Arrays.toString(b));
        } finally {
            pipe.close();
        }
    }

    /**
     * The original handoff scenario: many generates on one fixed-buffer pipeline with a sampling-config
     * swap in the middle. Greedy must remain reproducible after a creative detour on the reused plan,
     * and no demotion / exception may occur across the run.
     *
     * <p>The first two generations advance the reused DSP plans through eager, frozen, and captured
     * execution. Comparing the fresh generation directly with a later captured replay conflates that
     * lifecycle transition with sampling-config reuse: capture-safe cuBLAS selection can differ by ulps,
     * which greedy near-ties amplify into token changes. The assertion therefore compares captured
     * steady-state generations on both sides of a deterministic creative detour.</p>
     */
    @Test
    @DisplayName("fixed-buffer reuse survives a sampling-config swap (greedy→creative→greedy)")
    public void reuseSurvivesConfigSwap() throws Exception {
        GenerationPipeline pipe = fixedBufferPipeline();
        try {
            pipe.generate(PROMPT, N);                                // eager/freeze warmup
            pipe.generate(PROMPT, N);                                // capture-transition warmup
            int[] g1 = pipe.generate(PROMPT, N).getTokenIds();        // greedy, captured steady state

            pipe.setSamplingConfig(SamplingConfig.creative().toBuilder()
                    .seed(12345L)
                    .build());
            int[] creative = pipe.generate(PROMPT, N).getTokenIds();  // sampled, captured reuse
            assertTrue(creative.length > 0, "creative generate produced no tokens");

            pipe.setSamplingConfig(SamplingConfig.greedy());
            int[] g2 = pipe.generate(PROMPT, N).getTokenIds();        // greedy, captured reuse
            assertArrayEquals(g1, g2,
                    "steady-state greedy output changed after a creative detour on the reused plan.\n"
                    + "  g1=" + Arrays.toString(g1) + "\n  g2=" + Arrays.toString(g2));
        } finally {
            pipe.close();
        }
    }
}
