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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff.vlm;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.vlm.training.VlmTrainingExample;
import org.eclipse.deeplearning4j.vlm.training.VlmTrainingPipeline;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.VlmFineTuneConfig;
import org.nd4j.autodiff.samediff.config.VlmTrainingConfig;
import org.nd4j.autodiff.samediff.peft.PeftModel;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the VLM Training Pipeline.
 *
 * All tests use small synthetic SameDiff models — no VisionLanguageModel is loaded.
 * Tests verify config defaults, validation, data class, variable freezing,
 * LoRA application, and pipeline construction.
 *
 * @author Adam Gibson
 */
@Slf4j
@DisplayName("VLM Training Pipeline Tests")
public class TestVlmTrainingPipeline extends BaseNd4jTestWithBackends {

    private static final int TEST_IMAGE_SIZE = 224;
    private static final int TEST_PATCH_SIZE = 14;
    private static final int TEST_PATCHES_PER_SIDE = TEST_IMAGE_SIZE / TEST_PATCH_SIZE;
    private static final int TEST_IMAGE_TOKENS = TEST_PATCHES_PER_SIDE * TEST_PATCHES_PER_SIDE;

    @Override
    public char ordering() {
        return 'c';
    }

    // =========================================================================
    // 1. VlmTrainingConfig defaults
    // =========================================================================

    @Test
    @DisplayName("testVlmTrainingConfigDefaults - verify default field values")
    public void testVlmTrainingConfigDefaults() {
        VlmTrainingConfig config = VlmTrainingConfig.builder().build();

        assertEquals(2e-5,  config.getLearningRate(),    1e-10, "default learningRate");
        assertEquals(0.0,   config.getMinLearningRate(), 1e-10, "default minLearningRate");
        assertEquals(0.03,  config.getWarmupRatio(),     1e-10, "default warmupRatio");
        assertEquals(1,     config.getNumEpochs(),              "default numEpochs");
        assertEquals(2048,  config.getMaxSeqLength(),           "default maxSeqLength");
        assertEquals(8,     config.getGradientAccumulationSteps(), "default gradientAccumulationSteps");
        assertEquals(0.01,  config.getWeightDecay(),     1e-10, "default weightDecay");
        assertEquals(1.0,   config.getMaxGradNorm(),     1e-10, "default maxGradNorm");
        assertEquals(224,   config.getImageSize(),              "default imageSize");
        assertEquals(14,    config.getPatchSize(),              "default patchSize");
        assertEquals(DataType.BFLOAT16, config.getComputeDataType(), "default computeDataType");
        assertNull(config.getLossScaleConfig(), "default lossScaleConfig should be null");
        assertNotNull(config.getChatTemplate(), "default chatTemplate must not be null");
    }

    // =========================================================================
    // 2. VlmTrainingConfig validation
    // =========================================================================

    @Test
    @DisplayName("testVlmTrainingConfigValidation - invalid configs throw IllegalStateException")
    public void testVlmTrainingConfigValidation() {
        // Invalid: learningRate <= 0
        VlmTrainingConfig badLr = VlmTrainingConfig.builder()
                .learningRate(-1e-5).build();
        assertThrows(IllegalStateException.class, badLr::validate,
                "negative learningRate must fail validation");

        // Invalid: minLearningRate >= learningRate
        VlmTrainingConfig badMin = VlmTrainingConfig.builder()
                .learningRate(1e-5)
                .minLearningRate(2e-5)
                .build();
        assertThrows(IllegalStateException.class, badMin::validate,
                "minLearningRate >= learningRate must fail");

        // Invalid: numEpochs < 1
        VlmTrainingConfig badEpochs = VlmTrainingConfig.builder()
                .numEpochs(0).build();
        assertThrows(IllegalStateException.class, badEpochs::validate,
                "numEpochs=0 must fail");

        // Invalid: imageSize not divisible by patchSize
        VlmTrainingConfig badPatches = VlmTrainingConfig.builder()
                .imageSize(200)
                .patchSize(14)
                .build();
        assertThrows(IllegalStateException.class, badPatches::validate,
                "200 not divisible by 14 must fail");

        // Invalid: imageSize <= 0
        VlmTrainingConfig badSize = VlmTrainingConfig.builder()
                .imageSize(0).build();
        assertThrows(IllegalStateException.class, badSize::validate,
                "imageSize=0 must fail");

        // Invalid: patchSize <= 0
        VlmTrainingConfig badPatch = VlmTrainingConfig.builder()
                .patchSize(0).build();
        assertThrows(IllegalStateException.class, badPatch::validate,
                "patchSize=0 must fail");

        // Valid: default config
        VlmTrainingConfig valid = VlmTrainingConfig.builder().build();
        assertDoesNotThrow(valid::validate, "default config must pass validation");

        // Valid: imageSize=224, patchSize=14 (224 % 14 == 0)
        VlmTrainingConfig valid2 = VlmTrainingConfig.builder()
                .imageSize(224).patchSize(14).build();
        assertDoesNotThrow(valid2::validate);
    }

    // =========================================================================
    // 3. VlmTrainingConfig factory methods
    // =========================================================================

    @Test
    @DisplayName("testLoRATrainingConfig - verify loraTraining factory")
    public void testLoRATrainingConfig() {
        VlmTrainingConfig config = VlmTrainingConfig.loraTraining();
        assertDoesNotThrow(config::validate);
        assertEquals(2e-5, config.getLearningRate(), 1e-10);
        assertEquals(1, config.getNumEpochs());
        assertEquals(8, config.getGradientAccumulationSteps());
        assertEquals(DataType.BFLOAT16, config.getComputeDataType());
    }

    @Test
    @DisplayName("testFullFineTuneConfig - verify fullFineTune factory")
    public void testFullFineTuneConfig() {
        VlmTrainingConfig config = VlmTrainingConfig.fullFineTune();
        assertDoesNotThrow(config::validate);
        assertEquals(5e-6, config.getLearningRate(), 1e-10);
        assertEquals(1, config.getNumEpochs());
        assertEquals(16, config.getGradientAccumulationSteps());
        assertEquals(DataType.BFLOAT16, config.getComputeDataType());
    }

    // =========================================================================
    // 4. VlmTrainingExample data class
    // =========================================================================

    @Test
    @DisplayName("testVlmTrainingExampleCreation - verify data class builder and helpers")
    public void testVlmTrainingExampleCreation() {
        INDArray image = Nd4j.rand(DataType.FLOAT, 3, 224, 224);

        // With system message
        VlmTrainingExample ex = VlmTrainingExample.builder()
                .imageFeatures(image)
                .prompt("What do you see?")
                .response("A cat.")
                .systemMessage("You are a helpful assistant.")
                .build();

        assertNotNull(ex.getImageFeatures());
        assertEquals("What do you see?", ex.getPrompt());
        assertEquals("A cat.", ex.getResponse());
        assertEquals("You are a helpful assistant.", ex.getSystemMessage());
        assertTrue(ex.hasSystemMessage());
        assertFalse(ex.isPreEncoded(), "rank-3 image should not be pre-encoded");

        // Without system message
        VlmTrainingExample noSys = VlmTrainingExample.builder()
                .imageFeatures(image)
                .prompt("Describe this.")
                .response("It's an image.")
                .build();

        assertFalse(noSys.hasSystemMessage());
        assertNull(noSys.getSystemMessage());

        // Pre-encoded features (rank 2)
        INDArray encoded = Nd4j.rand(DataType.FLOAT, 256, 768);
        VlmTrainingExample preEnc = VlmTrainingExample.builder()
                .imageFeatures(encoded)
                .prompt("Summarize.")
                .response("Short summary.")
                .build();

        assertTrue(preEnc.isPreEncoded(), "rank-2 tensor should be identified as pre-encoded");
    }

    @Test
    @DisplayName("testVlmTrainingExampleNullPrompt - null prompt must throw NullPointerException")
    public void testVlmTrainingExampleNullPrompt() {
        INDArray image = Nd4j.rand(DataType.FLOAT, 3, 224, 224);
        assertThrows(NullPointerException.class, () ->
                VlmTrainingExample.builder()
                        .imageFeatures(image)
                        .prompt(null)
                        .response("A response.")
                        .build(),
                "null prompt must throw NullPointerException via @NonNull");
    }

    // =========================================================================
    // 5. Freeze vision encoder
    // =========================================================================

    @Test
    @DisplayName("testFreezeVisionEncoder - all VARIABLE vars in encoder become CONSTANT")
    public void testFreezeVisionEncoder() {
        SameDiff encoder = buildMinimalEncoder();
        SameDiff decoder = buildMinimalDecoder();

        // Count variables before pipeline creation
        long variablesBefore = encoder.variables().stream()
                .filter(v -> v.getVariableType() == VariableType.VARIABLE)
                .count();
        assertTrue(variablesBefore > 0, "encoder must have VARIABLE-type weights before freezing");

        VlmFineTuneConfig vlmConfig = withTestImageGeometry(
                VlmFineTuneConfig.loraOnly()); // freezeVisionEncoder=true
        VlmTrainingConfig trainingConfig = VlmTrainingConfig.builder().build();

        VlmTrainingPipeline pipeline = VlmTrainingPipeline.create(
                encoder, decoder, vlmConfig, trainingConfig);

        // After pipeline creation all encoder VARIABLE vars should be CONSTANT
        long variablesAfter = encoder.variables().stream()
                .filter(v -> v.getVariableType() == VariableType.VARIABLE)
                .count();
        assertEquals(0, variablesAfter,
                "All encoder VARIABLE vars must be CONSTANT after freezing");

        long constantsAfter = encoder.variables().stream()
                .filter(v -> v.getVariableType() == VariableType.CONSTANT)
                .count();
        assertEquals(variablesBefore, constantsAfter,
                "Frozen variables must appear as CONSTANTs");
    }

    // =========================================================================
    // 6. Train projector only — encoder frozen, decoder has no LoRA
    // =========================================================================

    @Test
    @DisplayName("testTrainProjectorOnly - encoder frozen, decoder has no LoRA adapter")
    public void testTrainProjectorOnly() {
        SameDiff encoder = buildMinimalEncoder();
        SameDiff decoder = buildMinimalDecoder();

        VlmFineTuneConfig vlmConfig = withTestImageGeometry(
                VlmFineTuneConfig.projectorOnly()); // no LoRA, frozen encoder
        VlmTrainingConfig trainingConfig = VlmTrainingConfig.projectorOnly();

        VlmTrainingPipeline pipeline = VlmTrainingPipeline.create(
                encoder, decoder, vlmConfig, trainingConfig);

        // Projector-only: no PEFT applied to decoder
        assertNull(pipeline.getPeftModel(),
                "projectorOnly config must not apply PEFT to the decoder");

        // Encoder should be frozen
        long encoderVariables = encoder.variables().stream()
                .filter(v -> v.getVariableType() == VariableType.VARIABLE)
                .count();
        assertEquals(0, encoderVariables, "Encoder must be frozen in projectorOnly config");
    }

    // =========================================================================
    // 7. LoRA on decoder
    // =========================================================================

    @Test
    @DisplayName("testLoRAOnDecoder - PeftModel wraps decoder when llmLoraConfig is set")
    public void testLoRAOnDecoder() {
        SameDiff encoder = buildMinimalEncoder();
        SameDiff decoder = buildMinimalDecoder();

        VlmFineTuneConfig vlmConfig = withTestImageGeometry(
                VlmFineTuneConfig.loraOnly()); // LoRA on LLM
        VlmTrainingConfig trainingConfig = VlmTrainingConfig.loraTraining();

        VlmTrainingPipeline pipeline = VlmTrainingPipeline.create(
                encoder, decoder, vlmConfig, trainingConfig);

        PeftModel peft = pipeline.getPeftModel();
        assertNotNull(peft, "PeftModel must be non-null when llmLoraConfig is set");

        // The trainable decoder is the PEFT-modified model
        assertNotNull(pipeline.getDecoderModel(),
                "getDecoderModel() must return the PEFT-modified SameDiff");

        // mergeAndExport must return a SameDiff (with adapters merged)
        SameDiff merged = pipeline.mergeAndExport();
        assertNotNull(merged, "mergeAndExport must return a non-null SameDiff");
    }

    // =========================================================================
    // 8. Pipeline creation factory
    // =========================================================================

    @Test
    @DisplayName("testPipelineCreation - factory method validates configs and wires components")
    public void testPipelineCreation() {
        SameDiff encoder = buildMinimalEncoder();
        SameDiff decoder = buildMinimalDecoder();

        VlmFineTuneConfig vlmConfig = withTestImageGeometry(VlmFineTuneConfig.loraOnly());
        VlmTrainingConfig trainingConfig = VlmTrainingConfig.loraTraining();

        VlmTrainingPipeline pipeline = VlmTrainingPipeline.create(
                encoder, decoder, vlmConfig, trainingConfig);

        assertNotNull(pipeline);
        assertEquals(vlmConfig, pipeline.getVlmConfig());
        assertEquals(trainingConfig, pipeline.getTrainingConfig());
        assertNotNull(pipeline.getDecoderModel());
    }

    // =========================================================================
    // 9. Full fine-tune config via VlmFineTuneConfig.fullFinetune()
    // =========================================================================

    @Test
    @DisplayName("testFullFineTuneVlmConfig - encoder not frozen, no LoRA")
    public void testFullFineTuneVlmConfig() {
        SameDiff encoder = buildMinimalEncoder();
        SameDiff decoder = buildMinimalDecoder();

        VlmFineTuneConfig vlmConfig = withTestImageGeometry(
                VlmFineTuneConfig.fullFinetune()); // not frozen, no LoRA
        VlmTrainingConfig trainingConfig = VlmTrainingConfig.fullFineTune();

        VlmTrainingPipeline pipeline = VlmTrainingPipeline.create(
                encoder, decoder, vlmConfig, trainingConfig);

        // Encoder must NOT be frozen
        long encoderVariables = encoder.variables().stream()
                .filter(v -> v.getVariableType() == VariableType.VARIABLE)
                .count();
        assertTrue(encoderVariables > 0,
                "Full fine-tune must leave encoder variables trainable (not frozen)");

        // No LoRA on decoder
        assertNull(pipeline.getPeftModel(),
                "fullFinetune config has no llmLoraConfig, so no PeftModel");
    }

    // =========================================================================
    // 10. computeNumPatches helper
    // =========================================================================

    @Test
    @DisplayName("testComputeNumPatches - imageSize/patchSize calculation")
    public void testComputeNumPatches() {
        VlmTrainingConfig config224 = VlmTrainingConfig.builder()
                .imageSize(224).patchSize(14).build();
        assertEquals(256, config224.computeNumPatches(),
                "(224/14)^2 = 256 patches");

        VlmTrainingConfig config336 = VlmTrainingConfig.builder()
                .imageSize(336).patchSize(14).build();
        assertEquals(576, config336.computeNumPatches(),
                "(336/14)^2 = 576 patches");
    }

    // =========================================================================
    // Helper builders for minimal synthetic SameDiff models
    // =========================================================================

    /**
     * Build a tiny vision encoder SameDiff with a single VARIABLE weight.
     */
    private static SameDiff buildMinimalEncoder() {
        SameDiff encoder = SameDiff.create();
        encoder.placeHolder("pixel_values", DataType.FLOAT, -1, 3, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE);
        encoder.var("vision_encoder/patch_embed/weight",
                Nd4j.randn(DataType.FLOAT, 64, 3, TEST_PATCH_SIZE, TEST_PATCH_SIZE));
        encoder.var("vision_encoder/patch_embed/bias",
                Nd4j.zeros(DataType.FLOAT, 64));
        return encoder;
    }

    private static VlmFineTuneConfig withTestImageGeometry(VlmFineTuneConfig config) {
        return VlmFineTuneConfig.builder()
                .freezeVisionEncoder(config.isFreezeVisionEncoder())
                .trainProjector(config.isTrainProjector())
                .shareWeights(config.isShareWeights())
                .llmLoraConfig(config.getLlmLoraConfig())
                .visionLoraConfig(config.getVisionLoraConfig())
                .imageResolution(TEST_IMAGE_SIZE)
                .patchSize(TEST_PATCH_SIZE)
                .maxImageTokens(TEST_IMAGE_TOKENS)
                .visionOutputVariable(config.getVisionOutputVariable())
                .projectorOutputVariable(config.getProjectorOutputVariable())
                .build();
    }

    /**
     * Build a tiny LLM decoder SameDiff with a 2-D weight suitable for LoRA injection.
     */
    private static SameDiff buildMinimalDecoder() {
        SameDiff decoder = SameDiff.create();
        decoder.placeHolder("input_ids", DataType.INT64, -1, -1);
        // 2-D weight — required shape for LoRA injection
        decoder.var("llm/q_proj", Nd4j.randn(DataType.FLOAT, 64, 64));
        decoder.var("llm/v_proj", Nd4j.randn(DataType.FLOAT, 64, 64));
        decoder.var("llm/embed",  Nd4j.randn(DataType.FLOAT, 100, 64));
        return decoder;
    }
}
