/*
 * TestMtpPredictorContract — graph-topology contract fixture for the bundled
 * NextN/MTP predictor exported by LLaMAArchitecture (T2a red fixture).
 *
 * Cases 1 and 2 assert the POST-NORM export contract on the built SameDiff
 * graph topology (CPU construction only — no model weights beyond tiny
 * synthetic tensors required by the builder, no GPU, no execution):
 *
 *   Case 1 (F1a) TARGET_EXPORT_POST_NORM:
 *     "target_hidden_states" must be an identity whose input is the OUTPUT of
 *     the "model.norm" RMSNorm op — not the raw last transformer block output.
 *     Current state: LLaMAArchitecture.java:220 builds the identity over the
 *     pre-norm hidden state; the norm is built afterwards at line 225.
 *
 *   Case 2 (F1b) RECURSIVE_CARRY_POST_NORM:
 *     "mtp_hidden_states" must be the identity over the "mtp.shared_head.norm"
 *     output, not the raw predictor block output. Current state:
 *     LLaMAArchitecture.java:359 exports the pre-norm predictor block output;
 *     "mtp.shared_head.norm" is built separately at lines 360-361.
 *
 * Cases 3 and 4 (BOOTSTRAP_NO_ZERO_ROW / IDS_SHIFTED_LEFT) would assert the
 * GenerationPipeline.prepareBundledMtp prefill bootstrap contract (no all-zero
 * hidden row; predictor prefill ids shifted left of the prompt ids). That
 * private method performs a live MTP-branch execution (outputWithSession) and
 * therefore requires the full pipeline harness plus a real executable model;
 * they are skipped here and tracked as T2b.
 */
package org.eclipse.deeplearning4j.llm.generation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
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

public class TestMtpPredictorContract {

    private static final String TARGET_HIDDEN_NAME = "target_hidden_states";
    private static final String MTP_HIDDEN_NAME = "mtp_hidden_states";
    private static final String TARGET_NORM_OUTPUT = "model.norm";
    private static final String MTP_SHARED_HEAD_NORM_OUTPUT = "mtp.shared_head.norm";

    private static final String GENERATION_PIPELINE_CLASS =
            "org.eclipse.deeplearning4j.llm.generation.GenerationPipeline";

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
    private SameDiff buildBundledMtpGraph() {
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
    private Map<String, INDArray> mtpWeights() {
        Map<String, INDArray> weights = new HashMap<>();
        weights.put("token_embd.weight",
                Nd4j.linspace(DataType.FLOAT, 0.0, 1.0, VOCAB_SIZE * HIDDEN_SIZE).reshape(VOCAB_SIZE, HIDDEN_SIZE));
        weights.put("output_norm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));
        for (int l = 0; l < TARGET_LAYERS; l++) {
            putTransformerBlockWeights(weights, "blk." + l);
        }
        // Bundled NextN predictor: transformer block at prefix "blk.<numLayers>"
        // plus the blk.<numLayers>.nextn.* payload tensors.
        putTransformerBlockWeights(weights, "blk." + TARGET_LAYERS);
        String nextnPrefix = "blk." + TARGET_LAYERS + ".nextn";
        weights.put(nextnPrefix + ".eh_proj.weight",
                Nd4j.linspace(DataType.FLOAT, 0.0, 1.0, 2 * HIDDEN_SIZE * HIDDEN_SIZE).reshape(2 * HIDDEN_SIZE, HIDDEN_SIZE));
        weights.put(nextnPrefix + ".enorm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));
        weights.put(nextnPrefix + ".hnorm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));
        weights.put(nextnPrefix + ".shared_head_norm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));
        return weights;
    }

    private void putTransformerBlockWeights(Map<String, INDArray> weights, String prefix) {
        weights.put(prefix + ".attn_norm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));
        weights.put(prefix + ".attn_q.weight", projection(HIDDEN_SIZE));
        weights.put(prefix + ".attn_k.weight", projection(HIDDEN_SIZE));
        weights.put(prefix + ".attn_v.weight", projection(HIDDEN_SIZE));
        weights.put(prefix + ".attn_output.weight", projection(HIDDEN_SIZE));
        weights.put(prefix + ".attn_q_norm.weight", Nd4j.ones(DataType.FLOAT, HEAD_DIM));
        weights.put(prefix + ".attn_k_norm.weight", Nd4j.ones(DataType.FLOAT, HEAD_DIM));
        weights.put(prefix + ".ffn_gate.weight", projection(INTERMEDIATE_SIZE));
        weights.put(prefix + ".ffn_up.weight", projection(INTERMEDIATE_SIZE));
        weights.put(prefix + ".ffn_down.weight", projection(HIDDEN_SIZE));
        weights.put(prefix + ".ffn_norm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));
    }

    private INDArray projection(int outDim) {
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
     * Cases 3+4 (BOOTSTRAP_NO_ZERO_ROW / IDS_SHIFTED_LEFT) live in
     * GenerationPipeline.prepareBundledMtp. The method is private and executes
     * the MTP branch (outputWithSession) plus creates an inference session, so
     * invoking it without a real model + pipeline harness is not possible.
     * Skip explicitly and defer to T2b.
     */
    @Test
    @DisplayName("F2a/F2b: prepareBundledMtp bootstrap contract requires the pipeline harness (T2b)")
    void bootstrapContractsRequirePipelineHarness() {
        boolean methodPresent = false;
        try {
            Class<?> pipelineClass = Class.forName(GENERATION_PIPELINE_CLASS);
            for (Method m : pipelineClass.getDeclaredMethods()) {
                if ("prepareBundledMtp".equals(m.getName())) {
                    methodPresent = true;
                    break;
                }
            }
        } catch (ClassNotFoundException e) {
            // samediff-llm not on the test classpath in this configuration.
        }
        Assumptions.assumeTrue(false,
                "SKIP [T2b]: prepareBundledMtp is " + (methodPresent ? "present on the classpath" : "not on the classpath")
                        + " but is private and performs a live MTP-branch execution (outputWithSession) plus session creation,"
                        + " so it cannot be driven without the pipeline harness + real model."
                        + " BOOTSTRAP_NO_ZERO_ROW (no all-zero hidden row) and IDS_SHIFTED_LEFT (predictor prefill ids"
                        + " shifted left of prompt ids) are deferred to the T2b harness.");
    }
}
