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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff.training;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.config.LoraConfig;
import org.nd4j.autodiff.samediff.config.LossScaleConfig;
import org.nd4j.autodiff.samediff.config.SFTConfig;
import org.nd4j.autodiff.samediff.config.TaskType;
import org.nd4j.autodiff.samediff.training.SFTTrainingPipeline;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.curation.format.ChatTemplate;
import org.nd4j.linalg.dataset.curation.format.ConversationExtension;
import org.nd4j.linalg.dataset.curation.format.ConversationTurn;
import org.nd4j.linalg.dataset.curation.format.FormattedExample;
import org.nd4j.linalg.dataset.curation.format.InstructionDataFormatter;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.AbstractMap;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link SFTTrainingPipeline} and {@link SFTConfig}.
 *
 * All tests use small synthetic SameDiff models to avoid requiring real LLM weights.
 * The goal is to verify:
 * <ul>
 *   <li>Pipeline construction and configuration defaults</li>
 *   <li>Correct response-token masking (the fundamental SFT invariant)</li>
 *   <li>Training loop integration with {@link SameDiff#fit}</li>
 *   <li>LoRA adapter wrapping through PEFT</li>
 *   <li>Multi-turn conversation synthesis via {@link ConversationExtension}</li>
 *   <li>Mixed-precision config propagation</li>
 *   <li>Input validation on bad configs</li>
 * </ul>
 *
 * @author Adam Gibson
 */
@Slf4j
@NativeTag
@Tag(TagNames.TRAINING)
@Tag(TagNames.SAMEDIFF)
@DisplayName("SFT Training Pipeline Tests")
public class TestSFTTrainingPipeline extends BaseNd4jTestWithBackends {

    // =========================================================================
    // Helpers for building tiny synthetic SameDiff models
    // =========================================================================

    /**
     * Build a tiny two-layer MLP SameDiff model that accepts {@code input_ids}
     * (treated as floats for gradient purposes in this test) and produces
     * {@code logits}.  The network is intentionally small so training runs fast.
     *
     * Shape: input [1, seqLen, hiddenDim] -> linear -> relu -> linear -> output [1, seqLen, vocabSize]
     */
    private static SameDiff buildTinyModel(int seqLen, int hiddenDim, int vocabSize) {
        SameDiff sd = SameDiff.create();

        // Placeholder: input token embeddings (float proxy for token IDs in tests)
        sd.placeHolder("input_ids", DataType.FLOAT, 1, seqLen, hiddenDim);

        // Weights
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).muli(0.02));
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, hiddenDim, vocabSize).muli(0.02));
        SDVariable b2 = sd.var("b2", Nd4j.zeros(DataType.FLOAT, vocabSize));

        // Forward
        SDVariable input   = sd.getVariable("input_ids");
        SDVariable hidden  = sd.nn.relu(sd.mmul(input, w1), 0);
        SDVariable logits  = sd.mmul(hidden, w2).add(b2);

        // Labels placeholder
        sd.placeHolder("labels", DataType.FLOAT, 1, seqLen, vocabSize);
        SDVariable labels = sd.getVariable("labels");

        // Simple MSE loss (cross-entropy would need softmax; MSE suffices for the test)
        SDVariable diff = logits.sub(labels);
        sd.loss.meanSquaredError("loss", labels, logits, null);

        return sd;
    }

    /**
     * Build a minimal (instruction, response) dataset as pairs.
     */
    private static List<Map.Entry<String, String>> buildSyntheticPairs(int n) {
        List<Map.Entry<String, String>> pairs = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            pairs.add(new AbstractMap.SimpleImmutableEntry<>(
                    "Instruction " + i + ": What is " + i + " + " + i + "?",
                    "Response " + i + ": The answer is " + (i + i) + "."));
        }
        return pairs;
    }

    // =========================================================================
    // 1. SFTConfig defaults
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("SFTConfig default factory produces expected field values")
    public void testSFTConfigDefaults(Nd4jBackend backend) {
        SFTConfig cfg = SFTConfig.defaultSFT();

        assertEquals(ChatTemplate.CHATML, cfg.getChatTemplate(), "default template should be CHATML");
        assertEquals(2e-5, cfg.getLearningRate(), 1e-10, "default lr should be 2e-5");
        assertEquals(0.0,  cfg.getMinLearningRate(), 1e-10, "default minLr should be 0");
        assertEquals(0.03, cfg.getWarmupRatio(), 1e-10, "default warmupRatio should be 0.03");
        assertEquals(3, cfg.getNumEpochs(), "default numEpochs should be 3");
        assertEquals(2048, cfg.getMaxSeqLength(), "default maxSeqLength should be 2048");
        assertEquals(4, cfg.getGradientAccumulationSteps(), "default gradAccum should be 4");
        assertEquals(DataType.BFLOAT16, cfg.getComputeDataType(), "default dtype should be BFLOAT16");
        assertEquals(0.01, cfg.getWeightDecay(), 1e-10, "default weightDecay should be 0.01");
        assertEquals(1.0, cfg.getMaxGradNorm(), 1e-10, "default maxGradNorm should be 1.0");
        assertNull(cfg.getPeftConfig(), "default peftConfig should be null");
        assertNull(cfg.getSystemMessage(), "default systemMessage should be null");
        assertFalse(cfg.isPackSequences(), "default packSequences should be false");
        assertFalse(cfg.isUsePaddingFree(), "default usePaddingFree should be false");
    }

    // =========================================================================
    // 2. SFTConfig validation
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("SFTConfig.validate() throws on invalid learningRate")
    public void testSFTConfigValidationNegativeLr(Nd4jBackend backend) {
        SFTConfig cfg = SFTConfig.builder().learningRate(-1.0).build();
        assertThrows(IllegalStateException.class, cfg::validate,
                "negative learningRate should fail validation");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("SFTConfig.validate() throws when minLr >= lr")
    public void testSFTConfigValidationMinLrGeLr(Nd4jBackend backend) {
        SFTConfig cfg = SFTConfig.builder()
                .learningRate(1e-5)
                .minLearningRate(1e-5)  // equal — invalid
                .build();
        assertThrows(IllegalStateException.class, cfg::validate,
                "minLearningRate == learningRate should fail validation");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("SFTConfig.validate() throws on numEpochs < 1")
    public void testSFTConfigValidationZeroEpochs(Nd4jBackend backend) {
        SFTConfig cfg = SFTConfig.builder().numEpochs(0).build();
        assertThrows(IllegalStateException.class, cfg::validate,
                "zero epochs should fail validation");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("SFTConfig.validate() throws on gradientAccumulationSteps < 1")
    public void testSFTConfigValidationZeroGradAccum(Nd4jBackend backend) {
        SFTConfig cfg = SFTConfig.builder().gradientAccumulationSteps(0).build();
        assertThrows(IllegalStateException.class, cfg::validate,
                "zero gradAccum should fail validation");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Valid SFTConfig.validate() succeeds without throwing")
    public void testSFTConfigValidationValid(Nd4jBackend backend) {
        SFTConfig cfg = SFTConfig.defaultSFT();
        assertDoesNotThrow(cfg::validate, "default config should pass validation");
    }

    // =========================================================================
    // 3. Response masking
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Loss mask correctly marks only assistant response tokens as trainable")
    public void testSFTResponseMasking(Nd4jBackend backend) {
        // Format a single-turn conversation and inspect the char-level mask
        InstructionDataFormatter formatter = InstructionDataFormatter.builder()
                .template(ChatTemplate.CHATML)
                .build();

        String instruction = "What is 2+2?";
        String response    = "The answer is 4.";
        FormattedExample ex = formatter.format(instruction, response);

        String text     = ex.getText();
        int[]  charMask = ex.getLossMask();

        assertEquals(text.length(), charMask.length,
                "charMask length must equal text length");

        // Char mask: 1 only inside assistant response content (not in prefix/suffix)
        // Verify there are some trainable chars and some non-trainable chars
        int trainableChars = 0;
        for (int m : charMask) trainableChars += m;

        assertTrue(trainableChars > 0,
                "should have at least some trainable (assistant response) chars");
        assertTrue(trainableChars < charMask.length,
                "should have some non-trainable (prompt/template) chars");

        // The trainable section must correspond to the response text
        // Find the response content in the formatted text
        int responseStart = text.indexOf(response);
        assertTrue(responseStart >= 0, "response text must appear in formatted output");

        for (int c = responseStart; c < responseStart + response.length(); c++) {
            assertEquals(1, charMask[c],
                    "char at position " + c + " inside response content must be trainable");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("convertCharMaskToTokenMask maps char mask correctly to token positions")
    public void testConvertCharMaskToTokenMask(Nd4jBackend backend) {
        // Build a char mask where chars 8..15 are trainable (tokens 2 and 3 at 4 chars/tok)
        int charLen = 20;
        int[] charMask = new int[charLen];
        Arrays.fill(charMask, 8, 16, 1); // chars 8-15 trainable

        int seqLen = 5; // 5 tokens * 4 chars/tok = 20 chars
        int[] tokenMask = SFTTrainingPipeline.convertCharMaskToTokenMask(charMask, seqLen, 4);

        assertEquals(seqLen, tokenMask.length);
        // Token 0: chars 0-3  -> all 0  -> not trainable
        assertEquals(0, tokenMask[0], "token 0 should be non-trainable");
        // Token 1: chars 4-7  -> all 0  -> not trainable
        assertEquals(0, tokenMask[1], "token 1 should be non-trainable");
        // Token 2: chars 8-11 -> all 1  -> trainable
        assertEquals(1, tokenMask[2], "token 2 should be trainable");
        // Token 3: chars 12-15 -> all 1 -> trainable
        assertEquals(1, tokenMask[3], "token 3 should be trainable");
        // Token 4: chars 16-19 -> all 0 -> not trainable
        assertEquals(0, tokenMask[4], "token 4 should be non-trainable");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("convertCharMaskToTokenMask handles seqLen > char mask coverage gracefully")
    public void testConvertCharMaskPadding(Nd4jBackend backend) {
        // Short text of 8 chars but requesting 10 tokens (40 chars worth)
        int[] charMask = new int[]{1, 1, 1, 1, 0, 0, 0, 0};
        int[] tokenMask = SFTTrainingPipeline.convertCharMaskToTokenMask(charMask, 10, 4);

        assertEquals(10, tokenMask.length);
        // Tokens beyond the charMask coverage must be non-trainable
        for (int t = 2; t < 10; t++) {
            assertEquals(0, tokenMask[t],
                    "token " + t + " beyond charMask should be non-trainable");
        }
    }

    // =========================================================================
    // 4. LoRA factory methods
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("SFTConfig.loraDefaults() produces valid LoRA config")
    public void testSFTWithLoRAConfig(Nd4jBackend backend) {
        SFTConfig cfg = SFTConfig.loraDefaults(16);
        assertNotNull(cfg.getPeftConfig(), "loraDefaults should set peftConfig");
        assertInstanceOf(LoraConfig.class, cfg.getPeftConfig(), "peftConfig should be LoraConfig");

        LoraConfig lora = (LoraConfig) cfg.getPeftConfig();
        assertEquals(16, lora.getR(), "rank should be 16");
        assertEquals(32, lora.getLoraAlpha(), "alpha should be 2*rank=32");
        assertEquals(TaskType.CAUSAL_LM, lora.getTaskType());
        assertDoesNotThrow(cfg::validate, "loraDefaults config should pass validation");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("SFTConfig.qloraDefaults() produces valid QLoRA config")
    public void testSFTWithQLoRAConfig(Nd4jBackend backend) {
        SFTConfig cfg = SFTConfig.qloraDefaults();
        assertNotNull(cfg.getPeftConfig(), "qloraDefaults should set peftConfig");
        // QLoraConfig extends LoraConfig, so it's also an instance of LoraConfig
        assertNotNull(cfg.getPeftConfig().getPeftType(), "peftType must not be null");
        assertDoesNotThrow(cfg::validate, "qloraDefaults config should pass validation");
    }

    // =========================================================================
    // 5. Pipeline construction (no-PEFT)
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("SFTTrainingPipeline constructs without PEFT (full fine-tune)")
    public void testBasicSFTPipelineConstruction(Nd4jBackend backend) {
        SameDiff model = buildTinyModel(8, 16, 32);
        SFTConfig cfg = SFTConfig.builder()
                .chatTemplate(ChatTemplate.CHATML)
                .numEpochs(1)
                .gradientAccumulationSteps(1)
                .learningRate(1e-4)
                .build();

        SFTTrainingPipeline pipeline = new SFTTrainingPipeline(model, cfg);

        assertNotNull(pipeline.getModel(), "pipeline model must not be null");
        assertNull(pipeline.getPeftModel(), "peftModel should be null when no PEFT config");
        assertSame(model, pipeline.getModel(), "model should be the base model");
    }

    // =========================================================================
    // 6. trainFromPairs — formatting and masking end-to-end
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("trainFromPairs: formatting produces non-empty datasets with correct mask structure")
    public void testSFTFromPairs(Nd4jBackend backend) {
        // We test the formatting/masking path without requiring a complete training loop,
        // because the tiny synthetic model is not a proper causal LM.
        // The test verifies that:
        //   (a) FormattedExamples are produced for each pair,
        //   (b) each example has both trainable and non-trainable characters,
        //   (c) convertCharMaskToTokenMask produces a valid mask.

        List<Map.Entry<String, String>> pairs = buildSyntheticPairs(5);

        InstructionDataFormatter formatter = InstructionDataFormatter.builder()
                .template(ChatTemplate.CHATML)
                .build();

        for (Map.Entry<String, String> pair : pairs) {
            FormattedExample ex = formatter.format(pair.getKey(), pair.getValue());

            assertNotNull(ex.getText(), "formatted text must not be null");
            assertFalse(ex.getText().isEmpty(), "formatted text must not be empty");
            assertEquals(ex.getText().length(), ex.getLossMask().length,
                    "charMask length must equal text length");

            int trainable = 0;
            for (int m : ex.getLossMask()) trainable += m;
            assertTrue(trainable > 0, "each example must have some trainable chars");

            // Token mask conversion
            int seqLen = Math.min(128, (ex.getText().length() + 3) / 4);
            int[] tokenMask = SFTTrainingPipeline.convertCharMaskToTokenMask(
                    ex.getLossMask(), seqLen);
            assertEquals(seqLen, tokenMask.length, "tokenMask length must equal seqLen");

            for (int m : tokenMask) {
                assertTrue(m == 0 || m == 1, "token mask values must be 0 or 1");
            }
        }
    }

    // =========================================================================
    // 7. Multi-turn conversation extension
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("ConversationExtension synthesises multi-turn conversations formatted with loss masks")
    public void testSFTWithConversationExtension(Nd4jBackend backend) {
        // Build single-turn pairs and extend into multi-turn conversations
        List<String> instructions = Arrays.asList(
                "What is the capital of France?",
                "Translate 'hello' to Spanish.",
                "What year was the Eiffel Tower built?"
        );
        List<String> responses = Arrays.asList(
                "Paris is the capital of France.",
                "Hola is the Spanish word for hello.",
                "The Eiffel Tower was built in 1889."
        );

        ConversationExtension extender = ConversationExtension.builder()
                .minTurns(1)
                .maxTurns(2)
                .extensionProbability(1.0)
                .maxTotalTokens(512)
                .seed(42)
                .build();

        List<ConversationTurn> turns = ConversationExtension.fromPairs(instructions, responses);
        List<FormattedExample> examples = extender.extendAndFormat(turns, ChatTemplate.CHATML);

        assertFalse(examples.isEmpty(), "should produce at least one formatted example");

        for (FormattedExample ex : examples) {
            assertNotNull(ex.getText());
            assertFalse(ex.getText().isEmpty());
            assertEquals(ex.getText().length(), ex.getLossMask().length);

            // Multi-turn conversations must still mask only assistant content
            int trainable = 0;
            for (int m : ex.getLossMask()) trainable += m;
            assertTrue(trainable > 0,
                    "multi-turn example must have some trainable (assistant) chars");
        }
    }

    // =========================================================================
    // 8. TrainingConfig construction
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("buildTrainingConfig produces correct compute dtype and grad accum settings")
    public void testBuildTrainingConfig(Nd4jBackend backend) {
        SameDiff model = buildTinyModel(8, 16, 32);
        SFTConfig cfg = SFTConfig.builder()
                .computeDataType(DataType.FLOAT)  // use FLOAT to avoid BF16 ops on CPU
                .numEpochs(1)
                .gradientAccumulationSteps(2)
                .learningRate(1e-4)
                .build();

        SFTTrainingPipeline pipeline = new SFTTrainingPipeline(model, cfg);
        org.nd4j.autodiff.samediff.TrainingConfig tc = pipeline.buildTrainingConfig(100);

        assertNotNull(tc, "TrainingConfig must not be null");
        assertNotNull(tc.getUpdater(), "updater must not be null");
        assertEquals(2, tc.getGradientAccumulationSteps(),
                "gradientAccumulationSteps must match config");
    }

    // =========================================================================
    // 9. Mixed precision config propagation
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("SFTConfig with BF16 propagates mixed-precision settings to TrainingConfig")
    public void testSFTWithMixedPrecision(Nd4jBackend backend) {
        SameDiff model = buildTinyModel(8, 16, 32);
        SFTConfig cfg = SFTConfig.builder()
                .computeDataType(DataType.BFLOAT16)
                .numEpochs(1)
                .gradientAccumulationSteps(1)
                .learningRate(1e-4)
                .build();
        assertDoesNotThrow(cfg::validate, "BF16 SFT config should be valid");

        SFTTrainingPipeline pipeline = new SFTTrainingPipeline(model, cfg);
        org.nd4j.autodiff.samediff.TrainingConfig tc = pipeline.buildTrainingConfig(10);

        // BF16 compute dtype should be reflected in the TrainingConfig
        assertEquals(DataType.BFLOAT16, tc.getComputeDataType(),
                "TrainingConfig computeDataType must be BFLOAT16");
        // BF16 does not require loss scaling
        assertFalse(tc.isLossScalingEnabled(),
                "BF16 should not require loss scaling by default");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("SFTConfig with FP16 auto-enables dynamic loss scaling")
    public void testSFTWithFP16AutoLossScaling(Nd4jBackend backend) {
        SameDiff model = buildTinyModel(8, 16, 32);
        SFTConfig cfg = SFTConfig.builder()
                .computeDataType(DataType.FLOAT16)
                .numEpochs(1)
                .gradientAccumulationSteps(1)
                .learningRate(1e-4)
                .build();
        assertDoesNotThrow(cfg::validate, "FP16 SFT config should be valid");

        SFTTrainingPipeline pipeline = new SFTTrainingPipeline(model, cfg);
        org.nd4j.autodiff.samediff.TrainingConfig tc = pipeline.buildTrainingConfig(10);

        assertTrue(tc.isLossScalingEnabled(),
                "FP16 should auto-enable dynamic loss scaling");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Explicit LossScaleConfig is forwarded to TrainingConfig")
    public void testExplicitLossScaleConfig(Nd4jBackend backend) {
        SameDiff model = buildTinyModel(8, 16, 32);
        LossScaleConfig lsc = LossScaleConfig.staticScaling(128.0);
        SFTConfig cfg = SFTConfig.builder()
                .computeDataType(DataType.FLOAT)
                .lossScaleConfig(lsc)
                .numEpochs(1)
                .gradientAccumulationSteps(1)
                .learningRate(1e-4)
                .build();

        SFTTrainingPipeline pipeline = new SFTTrainingPipeline(model, cfg);
        org.nd4j.autodiff.samediff.TrainingConfig tc = pipeline.buildTrainingConfig(10);

        assertTrue(tc.isLossScalingEnabled(),
                "explicit LossScaleConfig should enable loss scaling");
    }

    // =========================================================================
    // 10. System message propagation
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("System message is prepended to conversations and does not appear in loss mask")
    public void testSystemMessageNotTrainable(Nd4jBackend backend) {
        String sysMsg = "You are a helpful assistant.";
        InstructionDataFormatter formatter = InstructionDataFormatter.builder()
                .template(ChatTemplate.CHATML)
                .build();

        List<ConversationTurn> conv = new ArrayList<>();
        conv.add(ConversationTurn.system(sysMsg));
        conv.add(ConversationTurn.user("What is 2+2?"));
        conv.add(ConversationTurn.assistant("4"));

        FormattedExample ex = formatter.format(conv);
        String text = ex.getText();
        int[] mask  = ex.getLossMask();

        // The system message text must appear in the formatted output
        assertTrue(text.contains(sysMsg),
                "system message must appear in formatted text");

        // The system message characters must all be 0 in the char mask
        int sysStart = text.indexOf(sysMsg);
        for (int c = sysStart; c < sysStart + sysMsg.length(); c++) {
            assertEquals(0, mask[c],
                    "system message char at " + c + " must be non-trainable");
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
