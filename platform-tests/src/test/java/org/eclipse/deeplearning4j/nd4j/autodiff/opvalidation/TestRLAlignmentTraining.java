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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.RLAlignmentTrainer;
import org.nd4j.autodiff.samediff.config.*;
import org.nd4j.autodiff.samediff.rl.*;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for RL alignment training configs and trainers (DPO, KTO, ORPO, GRPO, PPO, DAPO, GSPO, DrGRPO).
 *
 * @author Adam Gibson
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("RL Alignment Training Tests")
public class TestRLAlignmentTraining {

    private static final int BATCH = 2;
    private static final int SEQ_LEN = 4;
    private static final int VOCAB = 10;
    private static final int HIDDEN = 8;

    /**
     * Build a simple 2-layer SameDiff model with 3D logits [batch, seqLen, vocab].
     * Returns the SameDiff graph with placeholder "input" and output "logits".
     *
     * The trainers expect logits to be 3D [batch, seq, vocab] — this matches
     * production LLM usage where each token position has its own distribution.
     * The model produces 2D hidden activations [batch, vocab], then expands and
     * tiles to [batch, seqLen, vocab] to match the token sequence length.
     */
    private SameDiff buildSimpleModel() {
        SameDiff sd = SameDiff.create();

        // Input: [batch, seqLen] token IDs — placeholder covers dynamic batch size
        // (trainers may concatenate chosen+rejected so runtime batch can be 2*BATCH)
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, SEQ_LEN);

        // Layer 1 weights: [seqLen, hidden]
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, SEQ_LEN, HIDDEN).mul(0.1));
        SDVariable b1 = sd.var("b1", Nd4j.zeros(DataType.FLOAT, HIDDEN));
        SDVariable hidden = sd.nn.relu(input.mmul(w1).add(b1), 0);

        // Layer 2 weights: [hidden, vocab]
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, HIDDEN, VOCAB).mul(0.1));
        SDVariable b2 = sd.var("b2", Nd4j.zeros(DataType.FLOAT, VOCAB));
        SDVariable logits2d = hidden.mmul(w2).add(b2);  // [batch, vocab]

        // Expand to [batch, 1, vocab] then tile to [batch, seqLen, vocab]
        // so that logits are 3D matching the token sequence dimension expected by trainers.
        SDVariable logitsExp = sd.expandDims("logits_exp", logits2d, 1);  // [batch, 1, vocab]
        sd.tile("logits", logitsExp, 1, SEQ_LEN, 1);                      // [batch, seqLen, vocab]

        return sd;
    }

    /**
     * Create toy input data for preference-based methods.
     */
    private Map<String, INDArray> createPreferenceInputs() {
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("input", Nd4j.rand(DataType.FLOAT, BATCH, SEQ_LEN));
        inputs.put("chosen", Nd4j.rand(DataType.FLOAT, BATCH, SEQ_LEN));
        inputs.put("rejected", Nd4j.rand(DataType.FLOAT, BATCH, SEQ_LEN));
        return inputs;
    }

    @Test
    @DisplayName("DPO - Standard Loss")
    public void testDPOStandardLoss() {
        SameDiff policy = buildSimpleModel();
        SameDiff reference = buildSimpleModel();

        DPOConfig config = DPOConfig.standard("logits", "chosen", "rejected");
        assertEquals(DPOConfig.DPOVariant.STANDARD, config.getVariant());
        assertEquals(0.1, config.getBeta());

        DPOTrainer trainer = new DPOTrainer(policy, reference, config);
        assertNotNull(trainer);

        Map<String, INDArray> inputs = createPreferenceInputs();
        double loss = trainer.trainStep(inputs);
        assertTrue(Double.isFinite(loss), "Loss should be finite, got: " + loss);
        assertTrue(loss > 0, "DPO standard loss should be positive, got: " + loss);
    }

    @Test
    @DisplayName("DPO - IPO Variant")
    public void testDPOIPOVariant() {
        SameDiff policy = buildSimpleModel();
        SameDiff reference = buildSimpleModel();

        DPOConfig config = DPOConfig.builder()
                .policyLogitVariable("logits")
                .chosenVariable("chosen")
                .rejectedVariable("rejected")
                .variant(DPOConfig.DPOVariant.IPO)
                .beta(0.1)
                .build();
        assertEquals(DPOConfig.DPOVariant.IPO, config.getVariant());
        assertEquals("DPO-IPO", config.getMethodName());

        DPOTrainer trainer = new DPOTrainer(policy, reference, config);
        Map<String, INDArray> inputs = createPreferenceInputs();
        double loss = trainer.trainStep(inputs);
        assertTrue(Double.isFinite(loss), "IPO loss should be finite, got: " + loss);
    }

    @Test
    @DisplayName("DPO - RDPO Variant with Label Smoothing")
    public void testDPORDPOVariant() {
        SameDiff policy = buildSimpleModel();
        SameDiff reference = buildSimpleModel();

        DPOConfig config = DPOConfig.builder()
                .policyLogitVariable("logits")
                .chosenVariable("chosen")
                .rejectedVariable("rejected")
                .variant(DPOConfig.DPOVariant.RDPO)
                .beta(0.1)
                .labelSmoothing(0.1)
                .build();
        assertEquals(DPOConfig.DPOVariant.RDPO, config.getVariant());
        assertEquals(0.1, config.getLabelSmoothing());
        assertEquals("DPO-RDPO", config.getMethodName());

        DPOTrainer trainer = new DPOTrainer(policy, reference, config);
        Map<String, INDArray> inputs = createPreferenceInputs();
        double loss = trainer.trainStep(inputs);
        assertTrue(Double.isFinite(loss), "RDPO loss should be finite, got: " + loss);
    }

    @Test
    @DisplayName("KTO Loss")
    public void testKTOLoss() {
        SameDiff policy = buildSimpleModel();
        SameDiff reference = buildSimpleModel();

        KTOConfig config = KTOConfig.standard("logits", "desirability");
        assertEquals("KTO", config.getMethodName());
        assertEquals(0.1, config.getBetaDesirable());
        assertEquals(0.1, config.getBetaUndesirable());

        KTOTrainer trainer = new KTOTrainer(policy, reference, config);
        assertNotNull(trainer);

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("input", Nd4j.rand(DataType.FLOAT, BATCH, SEQ_LEN));
        // Desirability labels: 1=good, 0=bad
        inputs.put("desirability", Nd4j.create(new double[]{1.0, 0.0}).reshape(BATCH));
        double loss = trainer.trainStep(inputs);
        assertTrue(Double.isFinite(loss), "KTO loss should be finite, got: " + loss);
    }

    @Test
    @DisplayName("ORPO Loss - No Reference Model")
    public void testORPOLoss() {
        SameDiff policy = buildSimpleModel();

        ORPOConfig config = ORPOConfig.standard("logits", "chosen", "rejected");
        assertEquals("ORPO", config.getMethodName());
        assertEquals(0.1, config.getOrpoLambda());
        assertFalse(config.isUseReferenceModel(), "ORPO should not use reference model");

        ORPOTrainer trainer = new ORPOTrainer(policy, config);
        assertNotNull(trainer);

        Map<String, INDArray> inputs = createPreferenceInputs();
        double loss = trainer.trainStep(inputs);
        assertTrue(Double.isFinite(loss), "ORPO loss should be finite, got: " + loss);
    }

    @Test
    @DisplayName("Reward Model - Bradley-Terry")
    public void testRewardModelBradleyTerry() {
        SameDiff policy = buildSimpleModel();

        RewardModelConfig config = RewardModelConfig.bradleyTerry(
                "logits", "chosen", "rejected", "logits");
        assertEquals("RewardModel-BRADLEY_TERRY", config.getMethodName());
        assertEquals(RewardModelConfig.RewardType.BRADLEY_TERRY, config.getRewardType());
        assertFalse(config.isUseReferenceModel());

        RewardModelTrainer trainer = new RewardModelTrainer(policy, config);
        assertNotNull(trainer);

        Map<String, INDArray> inputs = createPreferenceInputs();
        double loss = trainer.trainStep(inputs);
        assertTrue(Double.isFinite(loss), "Reward model loss should be finite, got: " + loss);
    }

    @Test
    @DisplayName("GRPO Config")
    public void testGRPOConfig() {
        GRPOConfig config = GRPOConfig.standard("logits", 16);
        assertEquals("GRPO", config.getMethodName());
        assertEquals(16, config.getGroupSize());
        assertEquals(0.2, config.getClipEpsilon());
        assertEquals(0.01, config.getKlPenalty());
        assertEquals(256, config.getMaxNewTokens());
        assertEquals("logits", config.getPolicyLogitVariable());
    }

    @Test
    @DisplayName("PPO Config")
    public void testPPOConfig() {
        PPOConfig config = PPOConfig.standard("logits", "value_head");
        assertEquals("PPO", config.getMethodName());
        assertEquals(0.2, config.getClipEpsilon());
        assertEquals(0.5, config.getValueLossCoeff());
        assertEquals(0.01, config.getEntropyCoeff());
        assertEquals(4, config.getPpoEpochs());
        assertEquals(0.95, config.getGaeLambda());
        assertEquals(256, config.getMaxNewTokens());
        assertEquals("value_head", config.getValueVariable());
    }

    @Test
    @DisplayName("DAPO Config - Asymmetric Clipping")
    public void testDAPOConfig() {
        DAPOConfig config = DAPOConfig.builder()
                .policyLogitVariable("logits")
                .clipEpsilonLow(0.1)
                .clipEpsilonHigh(0.28)
                .tokenLevelKL(true)
                .dynamicSampling(true)
                .overlongFiltering(true)
                .groupSize(8)
                .maxNewTokens(256)
                .build();
        assertEquals("DAPO", config.getMethodName());
        assertEquals(0.1, config.getClipEpsilonLow());
        assertEquals(0.28, config.getClipEpsilonHigh());
        assertTrue(config.isTokenLevelKL());
        assertTrue(config.isDynamicSampling());
        assertTrue(config.isOverlongFiltering());
        assertEquals(8, config.getGroupSize());
    }

    @Test
    @DisplayName("GSPO Config")
    public void testGSPOConfig() {
        GSPOConfig config = GSPOConfig.builder()
                .policyLogitVariable("logits")
                .stabilityCoeff(0.1)
                .importanceWeightedAdvantage(true)
                .groupSize(8)
                .clipEpsilon(0.2)
                .maxNewTokens(256)
                .build();
        assertEquals("GSPO", config.getMethodName());
        assertEquals(0.1, config.getStabilityCoeff());
        assertTrue(config.isImportanceWeightedAdvantage());
        assertEquals(8, config.getGroupSize());
        assertEquals(0.2, config.getClipEpsilon());
    }

    @Test
    @DisplayName("DrGRPO Config")
    public void testDrGRPOConfig() {
        DrGRPOConfig config = DrGRPOConfig.builder()
                .policyLogitVariable("logits")
                .lengthNormalization(true)
                .baselineSubtraction(true)
                .groupSize(8)
                .clipEpsilon(0.2)
                .maxNewTokens(256)
                .build();
        assertEquals("DrGRPO", config.getMethodName());
        assertTrue(config.isLengthNormalization());
        assertTrue(config.isBaselineSubtraction());
        assertEquals(8, config.getGroupSize());
        assertEquals(0.2, config.getClipEpsilon());
    }

    @Test
    @DisplayName("RL Alignment Config Validation")
    public void testRLAlignmentConfigValidation() {
        // Missing policyLogitVariable should throw
        assertThrows(IllegalStateException.class, () -> {
            DPOConfig config = DPOConfig.builder()
                    .chosenVariable("chosen")
                    .rejectedVariable("rejected")
                    .build();
            config.validate();
        });

        // Negative beta should throw
        assertThrows(IllegalStateException.class, () -> {
            DPOConfig config = DPOConfig.builder()
                    .policyLogitVariable("logits")
                    .chosenVariable("chosen")
                    .rejectedVariable("rejected")
                    .beta(-0.1)
                    .build();
            config.validate();
        });

        // Label smoothing out of range should throw
        assertThrows(IllegalStateException.class, () -> {
            DPOConfig config = DPOConfig.builder()
                    .policyLogitVariable("logits")
                    .chosenVariable("chosen")
                    .rejectedVariable("rejected")
                    .labelSmoothing(0.6)
                    .build();
            config.validate();
        });

        // GRPO groupSize < 2 should throw
        assertThrows(IllegalStateException.class, () -> {
            GRPOConfig config = GRPOConfig.builder()
                    .policyLogitVariable("logits")
                    .groupSize(1)
                    .build();
            config.validate();
        });

        // PPO missing valueVariable should throw
        assertThrows(IllegalStateException.class, () -> {
            PPOConfig config = PPOConfig.builder()
                    .policyLogitVariable("logits")
                    .build();
            config.validate();
        });
    }

    @Test
    @DisplayName("Composite Reward Function")
    public void testRewardFunctionComposite() {
        RuleBasedRewardFunction r1 = RuleBasedRewardFunction.constant(1.0);
        RuleBasedRewardFunction r2 = RuleBasedRewardFunction.constant(2.0);

        CompositeRewardFunction composite = new CompositeRewardFunction();
        composite.add(r1, 0.5);
        composite.add(r2, 0.5);

        INDArray prompts = Nd4j.zeros(3, 4);
        INDArray completions = Nd4j.zeros(3, 4);
        INDArray scores = composite.score(prompts, completions);

        assertNotNull(scores);
        assertEquals(3, scores.length());
        // 0.5 * 1.0 + 0.5 * 2.0 = 1.5
        for (int i = 0; i < 3; i++) {
            assertEquals(1.5, scores.getDouble(i), 1e-6,
                    "Composite score should be 1.5 for batch element " + i);
        }
    }

    @Test
    @DisplayName("Top-K Sampling Strategy Config")
    public void testSamplingStrategyConfig() {
        TopKSamplingStrategy strategy = TopKSamplingStrategy.builder()
                .topK(50)
                .topP(0.9)
                .temperature(0.8)
                .seed(42)
                .build();

        assertEquals(50, strategy.getTopK());
        assertEquals(0.9, strategy.getTopP(), 1e-9);
        assertEquals(0.8, strategy.getTemperature(), 1e-9);
        assertEquals(42, strategy.getSeed());
    }
}