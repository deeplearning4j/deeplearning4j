/*
 * TestMtpPredictorContract — graph-topology + bootstrap contract fixture for
 * the bundled NextN/MTP predictor exported by LLaMAArchitecture (T2a/T2b).
 *
 * Cases 1 and 2 assert the POST-NORM export contract on the built SameDiff
 * graph topology (CPU construction only — no model weights beyond tiny
 * synthetic tensors required by the builder, no GPU, no execution):
 *
 *   Case 1 (F1a) TARGET_EXPORT_POST_NORM:
 *     "target_hidden_states" must be an identity whose input is the OUTPUT of
 *     the "model.norm" RMSNorm op — not the raw last transformer block output.
 *
 *   Case 2 (F1b) RECURSIVE_CARRY_POST_NORM:
 *     "mtp_hidden_states" must be the identity over the "mtp.shared_head.norm"
 *     output, not the raw predictor block output.
 *
 * Cases 3 and 4 (BOOTSTRAP_NO_ZERO_ROW / IDS_SHIFTED_LEFT) drive the private
 * GenerationPipeline.prepareBundledMtp through the production path: a tiny
 * CPU-executable bundled-MTP decoder graph, a pipeline instance built through
 * the same private constructor the production create() path uses, and the
 * graph's own InferenceSession — the exact wiring GenerationPipeline
 * establishes internally. The bootstrap contract is asserted on what the
 * method commits into the predictor KV cache and retains in its input maps:
 *
 *   Case 3 (F2a) BOOTSTRAP_NO_ZERO_ROW: the committed prefill rows contain NO
 *     all-zero row — row t is the (x_(t+1), h_t) pair, so the prefill cache
 *     holds exactly N meaningful rows (the warmup pair overwrites the last
 *     row's continuation at cache slot N).
 *
 *   Case 4 (F2b) IDS_SHIFTED_LEFT: the predictor prefill ids are the prompt
 *     ids shifted LEFT by one (x1..xN-1) with the first target-sampled token
 *     in the final slot (vLLM set_inputs_first_pass contract).
 */
package org.eclipse.deeplearning4j.llm.generation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.ggml.architecture.ArchitectureConfig;
import org.nd4j.ggml.architecture.LLaMAArchitecture;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class TestMtpPredictorContract {

    private static final String TARGET_HIDDEN_NAME = "target_hidden_states";
    private static final String MTP_HIDDEN_NAME = "mtp_hidden_states";
    private static final String TARGET_NORM_OUTPUT = "model.norm";
    private static final String MTP_SHARED_HEAD_NORM_OUTPUT = "mtp.shared_head.norm";

    private static final int TARGET_LAYERS = 1;
    private static final int HIDDEN_SIZE = 8;
    private static final int INTERMEDIATE_SIZE = 16;
    private static final int VOCAB_SIZE = 32;
    private static final int NUM_HEADS = 2;
    private static final int NUM_KV_HEADS = 2;
    private static final int HEAD_DIM = 4;

    /**
     * Builds the same bundled-MTP graph construction path LLaMAArchitecture
     * uses for real Qwen3.5-style GGUFs, at tiny scale. numLayers(1) with
     * numMtpLayers(1) makes layer index 1 the predictor block, so the MTP
     * branch construction under test (buildMtpBranch) runs. Only topology is
     * inspected — the graph is never executed.
     */
    private static SameDiff buildBundledMtpGraph() {
        LLaMAArchitecture arch = new LLaMAArchitecture() {
            @Override
            public ArchitectureConfig getConfig(GGMLMetadata metadata) {
                return ArchitectureConfig.builder()
                        .numLayers(TARGET_LAYERS)
                        .numMtpLayers(1)
                        .hiddenSize(HIDDEN_SIZE)
                        .intermediateSize(INTERMEDIATE_SIZE)
                        .numAttentionHeads(NUM_HEADS)
                        .numKVHeads(NUM_KV_HEADS)
                        .vocabSize(VOCAB_SIZE)
                        .contextLength(16)
                        .headDim(HEAD_DIM)
                        .build();
            }
        };
        return arch.buildGraph(
                GGMLMetadata.builder().architecture("llama").build(),
                mtpWeights(),
                ConversionOptions.builder()
                        .targetDataType(DataType.FLOAT)
                        .lastPositionLogitsOnly(true)
                        .build());
    }

    /** All tensors the target trunk, the head and buildMtpBranch require. */
    private static Map<String, INDArray> mtpWeights() {
        Map<String, INDArray> weights = new HashMap<>();
        weights.put("token_embd.weight",
                Nd4j.linspace(DataType.FLOAT, 0.0, 1.0, VOCAB_SIZE * HIDDEN_SIZE).reshape(VOCAB_SIZE, HIDDEN_SIZE));
        weights.put("output_norm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));
        for (int l = 0; l < TARGET_LAYERS; l++) {
            putTransformerBlockWeights(weights, "blk." + l);
        }
        // Bundled NextN predictor: transformer block at prefix "blk.<numLayers>"
        // plus the blk.<numLayers>.nextn.* payload tensors. The predictor block
        // mirrors the real Qwen3.5 bundled layout — a GATED attention layer, so
        // attn_q carries Q + gate interleaved per head ([2*heads*headDim, hidden]).
        putGatedTransformerBlockWeights(weights, "blk." + TARGET_LAYERS);
        String nextnPrefix = "blk." + TARGET_LAYERS + ".nextn";
        weights.put(nextnPrefix + ".eh_proj.weight",
                // [hidden, 2*hidden] — same layout as the real 27B tensor
                // (mtp.fc.weight = [5120, 10240]): concat(embed, target) @ W.
                Nd4j.linspace(DataType.FLOAT, 0.0, 1.0, HIDDEN_SIZE * 2 * HIDDEN_SIZE).reshape(HIDDEN_SIZE, 2 * HIDDEN_SIZE));
        weights.put(nextnPrefix + ".enorm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));
        weights.put(nextnPrefix + ".hnorm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));
        weights.put(nextnPrefix + ".shared_head_norm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));
        return weights;
    }

    private static void putTransformerBlockWeights(Map<String, INDArray> weights, String prefix) {
        weights.put(prefix + ".attn_norm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));
        weights.put(prefix + ".attn_q.weight", projection(HIDDEN_SIZE));
        weights.put(prefix + ".attn_k.weight", projection(HIDDEN_SIZE));
        weights.put(prefix + ".attn_v.weight", projection(HIDDEN_SIZE));
        weights.put(prefix + ".attn_output.weight", projection(HIDDEN_SIZE));
        weights.put(prefix + ".attn_q_norm.weight", Nd4j.ones(DataType.FLOAT, HEAD_DIM));
        weights.put(prefix + ".attn_k_norm.weight", Nd4j.ones(DataType.FLOAT, HEAD_DIM));
        weights.put(prefix + ".ffn_gate.weight", projection(INTERMEDIATE_SIZE));
        weights.put(prefix + ".ffn_up.weight", projection(INTERMEDIATE_SIZE));
        // GGUF layout: ffn_down is [hidden, intermediate] (down = swiglu @ W).
        weights.put(prefix + ".ffn_down.weight",
                Nd4j.linspace(DataType.FLOAT, 0.0, 1.0, HIDDEN_SIZE * INTERMEDIATE_SIZE).reshape(HIDDEN_SIZE, INTERMEDIATE_SIZE));
        weights.put(prefix + ".ffn_norm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));
    }

    /**
     * Gated-attention block layout (Qwen3.5 style): attn_q packs Q and gate
     * interleaved per head, so its output dim is 2 * numHeads * headDim. This is
     * what routes buildTransformerBlock through buildGatedAttention at execution.
     */
    private static void putGatedTransformerBlockWeights(Map<String, INDArray> weights, String prefix) {
        weights.put(prefix + ".attn_norm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));
        weights.put(prefix + ".attn_q.weight", projection(2 * NUM_HEADS * HEAD_DIM));
        weights.put(prefix + ".attn_k.weight", projection(NUM_KV_HEADS * HEAD_DIM));
        weights.put(prefix + ".attn_v.weight", projection(NUM_KV_HEADS * HEAD_DIM));
        weights.put(prefix + ".attn_output.weight", projection(HIDDEN_SIZE));
        weights.put(prefix + ".attn_q_norm.weight", Nd4j.ones(DataType.FLOAT, HEAD_DIM));
        weights.put(prefix + ".attn_k_norm.weight", Nd4j.ones(DataType.FLOAT, HEAD_DIM));
        weights.put(prefix + ".ffn_gate.weight", projection(INTERMEDIATE_SIZE));
        weights.put(prefix + ".ffn_up.weight", projection(INTERMEDIATE_SIZE));
        // GGUF layout: ffn_down is [hidden, intermediate] (down = swiglu @ W).
        weights.put(prefix + ".ffn_down.weight",
                Nd4j.linspace(DataType.FLOAT, 0.0, 1.0, HIDDEN_SIZE * INTERMEDIATE_SIZE).reshape(HIDDEN_SIZE, INTERMEDIATE_SIZE));
        weights.put(prefix + ".ffn_norm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));
    }

    private static INDArray projection(int outDim) {
        return Nd4j.linspace(DataType.FLOAT, 0.0, 1.0, outDim * HIDDEN_SIZE).reshape(outDim, HIDDEN_SIZE);
    }

    private SameDiffOp opProducing(SameDiff graph, String variableName) {
        for (SameDiffOp op : graph.getOps().values()) {
            if (op.getOutputsOfOp() != null && op.getOutputsOfOp().contains(variableName)) {
                return op;
            }
        }
        return null;
    }

    private void assertIdentityInputIsRmsNormOutput(SameDiff graph, String identityName, String normOutputName) {
        SDVariable identity = graph.getVariable(identityName);
        assertNotNull(identity, identityName + " must be exported by the bundled-MTP graph");
        assertEquals(identityName, identity.name(),
                "expected variable '" + identityName + "', found '" + identity.name() + "'");

        SameDiffOp identityOp = opProducing(graph, identityName);
        assertNotNull(identityOp, identityName + " must have a defining op");
        assertTrue(identityOp.getOp() != null && "identity".equals(identityOp.getOp().opName()),
                identityName + " must be produced by an identity op, found '"
                        + (identityOp.getOp() == null ? "null" : identityOp.getOp().opName()) + "'");
        assertNotNull(identityOp.getInputsToOp(), identityName + " identity op must have inputs");
        assertEquals(1, identityOp.getInputsToOp().size(),
                identityName + " identity op must have exactly one input, got " + identityOp.getInputsToOp());
        String identityInput = identityOp.getInputsToOp().get(0);

        SameDiffOp normOp = opProducing(graph, normOutputName);
        assertNotNull(normOp, normOutputName + " variable must exist in the graph");
        // buildRMSNorm composes mul/mean/sqrt/div and names the final scale
        // multiply with the output name, so the RMSNorm output is produced by
        // a 'multiply' op whose operands are the normalized activations and
        // the norm gamma.
        assertTrue(normOp.getOp() != null && "multiply".equals(normOp.getOp().opName()),
                normOutputName + " must be produced by the RMSNorm scale multiply, found '"
                        + (normOp.getOp() == null ? "null" : normOp.getOp().opName()) + "'");

        assertTrue(normOutputName.equals(identityInput),
                "[F1] " + identityName + " must wrap the post-norm hidden state: expected identity input '"
                        + normOutputName + "' but the identity consumes '" + identityInput
                        + "' (pre-norm hidden state). Defect evidence for the bundled-MTP export contract.");
    }

    @Test
    @DisplayName("F1a: target_hidden_states identity must consume the model.norm RMSNorm output")
    void targetExportMustBePostNorm() {
        SameDiff graph = buildBundledMtpGraph();
        assertIdentityInputIsRmsNormOutput(graph, TARGET_HIDDEN_NAME, TARGET_NORM_OUTPUT);
    }

    @Test
    @DisplayName("F1b: mtp_hidden_states identity must consume the mtp.shared_head.norm output")
    void recursiveCarryMustBePostNorm() {
        SameDiff graph = buildBundledMtpGraph();
        assertIdentityInputIsRmsNormOutput(graph, MTP_HIDDEN_NAME, MTP_SHARED_HEAD_NORM_OUTPUT);
    }

    /**
     * Cases 3+4 (BOOTSTRAP_NO_ZERO_ROW / IDS_SHIFTED_LEFT) drive
     * GenerationPipeline.prepareBundledMtp through the production path: a tiny
     * CPU-executable bundled-MTP decoder graph, a pipeline built through the same
     * private constructor GenerationPipeline.create() uses, and the graph's own
     * InferenceSession (the exact wiring the pipeline establishes internally).
     * Contract is asserted on the retained prefill input maps plus the freshly
     * committed predictor K/V cache.
     */
    @Test
    @DisplayName("F2a: prepareBundledMtp commits N shifted rows with NO all-zero bootstrap row")
    void bootstrapCommitsNoZeroRow() throws Exception {
        MtpHarness harness = new MtpHarness();
        try {
            INDArray hiddens = harness.prefillHiddens();
            int N = harness.prefillSeqLen;

            for (int t = 0; t < N; t++) {
                double rowMax = hiddens.get(NDArrayIndex.point(0), NDArrayIndex.point(t),
                        NDArrayIndex.all()).maxNumber().doubleValue();
                assertTrue(Math.abs(rowMax) > 0.0,
                        "[F2a] prefill hidden row " + t + " is all-zero: the bootstrap still "
                                + "inserts the junk zero row (rows must be (x_(t+1), h_t) pairs, "
                                + "cache = exactly " + N + " meaningful rows)");
            }

            // The predictor K/V cache must hold exactly N+1 meaningful rows: rows
            // 0..N-1 = the shifted prefill pairs (x_(t+1), h_t), and row N (= the
            // warmup cache slot firstDecodePos == N) = the (firstGen, h_(N-1)) pair.
            // Anything beyond N+1 nonzero would mean the warmup pair was APPENDED
            // behind a junk zero row instead of replacing the last shifted slot.
            INDArray keyCache = harness.keyCache();
            assertNotNull(keyCache, "[F2a] predictor K cache was not created");
            assertEquals(harness.maxKvLen, keyCache.size(1),
                    "[F2a] predictor K cache must be allocated for the production maxKvLen envelope");
            for (int t = 0; t <= N; t++) {
                double keyRowMax = keyCache.get(NDArrayIndex.all(), NDArrayIndex.point(t),
                        NDArrayIndex.all(), NDArrayIndex.all()).maxNumber().doubleValue();
                assertTrue(keyRowMax != 0.0,
                        "[F2a] predictor K cache row " + t + " is zero after prepareBundledMtp"
                                + (t < N ? ": the shifted prefill rows were not all committed"
                                         : ": the warmup pair must commit (firstGen, h_(N-1)) at slot N"));
            }
            for (int t = N + 1; t < harness.maxKvLen; t++) {
                double keyRowMax = keyCache.get(NDArrayIndex.all(), NDArrayIndex.point(t),
                        NDArrayIndex.all(), NDArrayIndex.all()).maxNumber().doubleValue();
                assertEquals(0.0, keyRowMax,
                        "[F2a] predictor K cache row " + t + " must stay untouched: the bootstrap"
                                + " committed more than the N shifted prefill rows + 1 warmup row");
            }
        } finally {
            harness.close();
        }
    }

    @Test
    @DisplayName("F2b: prepareBundledMtp feeds predictor ids shifted LEFT of the prompt ids")
    void bootstrapIdsShiftedLeft() throws Exception {
        MtpHarness harness = new MtpHarness();
        try {
            INDArray ids = harness.prefillIds();
            int[] prompt = harness.promptIds;
            int N = harness.prefillSeqLen;

            for (int t = 0; t < N - 1; t++) {
                assertEquals((long) prompt[t + 1], ids.getLong(0, t),
                        "[F2b] predictor prefill id at row " + t + " must be prompt id x_"
                                + (t + 1) + " (ids shifted LEFT of the prompt; vLLM contract)");
            }
            assertEquals((long) harness.firstTokenId, ids.getLong(0, N - 1),
                    "[F2b] final predictor prefill id must be the first target-sampled token");
        } finally {
            harness.close();
        }
    }

    /**
     * Drives prepareBundledMtp with the production wiring: tiny bundled-MTP decoder
     * graph (CPU execution), pipeline built via GenerationPipeline's private
     * constructor (the same 15-arg shape create() uses), and a pre-populated
     * InGraphKvState reuseState whose session/inputs/KV maps the method adopts —
     * the exact adoption path prepareBundledMtp implements for session reuse.
     * prepareBundledMtp is invoked reflectively with synthetic target prefill/warmup
     * hiddens; the adopted maps then carry the assertion surface for F2a/F2b.
     *
     * <p>On a CPU harness the method legitimately terminates at its final gate
     * ("Native MTP DSP plan handle is unavailable") AFTER both the prefill and the
     * scalar warmup executions completed and committed their state — that gate is
     * the CUDA/DSP infrastructure precondition, not part of the bootstrap contract.
     * The harness treats only that exact terminal as success; any earlier failure
     * (mask build, prefill execution, cache commit, warmup execution) is rethrown.
     * The '[MTP] Scalar warmup complete' log line in the run output is the
     * human-verifiable marker that both executions ran.</p>
     */
    private static final class MtpHarness implements AutoCloseable {
        static final String EXPECTED_CPU_TERMINAL = "Native MTP DSP plan handle is unavailable";

        final SameDiff decoder;
        final Object pipeline;
        final InGraphKvState reuseState;
        final int prefillSeqLen;
        final long maxKvLen;
        final int firstTokenId;
        final int[] promptIds;

        MtpHarness() throws Exception {
            this.decoder = buildBundledMtpGraph();
            this.pipeline = newPipeline(decoder);

            this.promptIds = new int[]{11, 23, 7, 19, 3};
            this.prefillSeqLen = promptIds.length;
            int actualPrefillLen = prefillSeqLen;
            this.maxKvLen = prefillSeqLen + 8L; // + decode budget, mirrors pipeline sizing
            this.firstTokenId = 29;
            int secondTokenId = 5;

            // The reuseState adoption path makes prepareBundledMtp write into THESE
            // maps, so the committed bootstrap state stays assertable regardless of
            // where the method returns.
            this.reuseState = new InGraphKvState();
            reuseState.mtpSession = decoder.getInferenceFactory().create(decoder);
            reuseState.mtpPrefillInputMap = new LinkedHashMap<>();
            reuseState.mtpKvBuffers = new LinkedHashMap<>();

            // Synthetic target prefill/warmup hiddens: deterministic per-row values so
            // a zero row (F2a) or a misaligned shift (F2b) is detectable exactly.
            INDArray targetPrefillHidden = Nd4j.zeros(DataType.FLOAT, 1, prefillSeqLen, HIDDEN_SIZE);
            for (int t = 0; t < prefillSeqLen; t++) {
                targetPrefillHidden.get(NDArrayIndex.point(0), NDArrayIndex.point(t),
                        NDArrayIndex.all()).assign(Nd4j.valueArrayOf(new long[]{HIDDEN_SIZE}, 1.0f + t));
            }
            INDArray targetWarmupHidden = Nd4j.zeros(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
            targetWarmupHidden.get(NDArrayIndex.point(0), NDArrayIndex.point(0),
                    NDArrayIndex.all()).assign(Nd4j.valueArrayOf(new long[]{HIDDEN_SIZE}, 99.0f));

            Method prepare = resolvePrepareMethod();
            Object[] args = new Object[]{
                    reuseState,
                    promptIds,
                    prefillSeqLen,
                    actualPrefillLen,
                    maxKvLen,
                    /* firstDecodePos */ prefillSeqLen,
                    firstTokenId,
                    secondTokenId,
                    targetPrefillHidden,
                    targetWarmupHidden};
            prepare.setAccessible(true);
            try {
                prepare.invoke(pipeline, args);
                // Reached only when a native DSP plan exists (CUDA harness): the
                // bootstrap completed fully and the maps below carry the contract.
            } catch (InvocationTargetException e) {
                Throwable cause = e.getCause();
                if (cause instanceof IllegalStateException
                        && cause.getMessage() != null
                        && cause.getMessage().contains(EXPECTED_CPU_TERMINAL)) {
                    // Expected CPU-harness terminal AFTER both executions committed.
                } else {
                    throw cause instanceof Exception ? (Exception) cause : e;
                }
            }
        }

        /**
         * Builds the pipeline instance exactly the way the production create() path
         * does: private 15-arg constructor, a stub Tokenizer proxy, discovered
         * ModelIOConfig, DSP disabled (CPU harness — no native plans), and a
         * GenerationPipelineConfig carrying the same maxKv envelope the assertions
         * expect.
         */
        private static Object newPipeline(SameDiff decoder) throws Exception {
            ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
            GenerationPipelineConfig config = GenerationPipelineConfig.builder()
                    .decoder(decoder)
                    .tokenizer(stubTokenizer())
                    .ioConfig(ioConfig)
                    .samplingConfig(SamplingConfig.greedy())
                    .maxPrefillLength(0) // dynamic prefill: buildInGraphCausalMask path
                    .dspEnabled(false)
                    .build();
            Constructor<?> constructor = GenerationPipeline.class.getDeclaredConstructors()[0];
            constructor.setAccessible(true);
            return constructor.newInstance(
                    decoder, false, null, false, stubTokenizer(), null,
                    ioConfig, null, 0L, null, null, null, false, config, null);
        }

        private static Tokenizer stubTokenizer() {
            return (Tokenizer) java.lang.reflect.Proxy.newProxyInstance(
                    Tokenizer.class.getClassLoader(), new Class<?>[]{Tokenizer.class},
                    (proxy, method, args) -> {
                        switch (method.getName()) {
                            case "getSpecialTokenIds": return java.util.Collections.emptySet();
                            case "getAddedTokens": return java.util.Collections.emptyMap();
                            default: throw new AssertionError(
                                    "Unexpected tokenizer access: " + method.getName());
                        }
                    });
        }

        private Method resolvePrepareMethod() throws Exception {
            for (Method m : GenerationPipeline.class.getDeclaredMethods()) {
                if (!"prepareBundledMtp".equals(m.getName())) continue;
                Class<?>[] p = m.getParameterTypes();
                if (p.length == 10 && p[0] == InGraphKvState.class && p[1] == int[].class
                        && p[2] == int.class && p[4] == long.class && p[5] == int.class
                        && p[8] == INDArray.class && p[9] == INDArray.class) {
                    return m;
                }
            }
            throw new NoSuchMethodException("GenerationPipeline.prepareBundledMtp(10-arg production signature)");
        }

        INDArray prefillIds() {
            return reuseState.mtpPrefillInputMap.get("mtp_input_ids");
        }

        INDArray prefillHiddens() {
            return reuseState.mtpPrefillInputMap.get("mtp_target_hidden_states");
        }

        INDArray keyCache() {
            return reuseState.mtpKvBuffers.get("mtp_past_key_values.0.key");
        }

        @Override
        public void close() {
            try {
                reuseState.close();
            } catch (Exception ignored) {
                // state teardown is best-effort in the harness
            }
            decoder.close();
        }
    }
}
