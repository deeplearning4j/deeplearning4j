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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.LoraConfig;
import org.nd4j.autodiff.samediff.config.VlmFineTuneConfig;
import org.nd4j.autodiff.samediff.config.VlmGRPOConfig;
import org.nd4j.autodiff.samediff.rl.GRPOTrainer;
import org.nd4j.autodiff.samediff.rl.RewardFunction;
import org.nd4j.autodiff.samediff.rl.SamplingStrategy;
import org.nd4j.autodiff.samediff.rl.VlmGRPOTrainer;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.VisionEncodePatches;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for VLM (Vision-Language Model) fine-tuning and VLM GRPO training support.
 *
 * Covers:
 *  - VlmFineTuneConfig creation, factory methods, and validation
 *  - VlmGRPOConfig creation, extension, and JSON round-trip
 *  - VisionEncodePatches op (single image and batched)
 *  - VlmGRPOTrainer construction and vision variable freezing
 *  - shareWeights configuration flag
 *
 * @author Adam Gibson
 */
@Slf4j
@DisplayName("VLM Fine-Tuning Tests")
public class TestVlmFinetuning extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    // =========================================================================
    // 1. VlmFineTuneConfig creation
    // =========================================================================

    @Test
    @DisplayName("testVlmFineTuneConfigCreation - loraOnly and fullFinetune factory methods")
    public void testVlmFineTuneConfigCreation() {
        // loraOnly()
        VlmFineTuneConfig loraOnly = VlmFineTuneConfig.loraOnly();
        assertTrue(loraOnly.isFreezeVisionEncoder(),
                "loraOnly should freeze the vision encoder");
        assertTrue(loraOnly.isTrainProjector(),
                "loraOnly should train the projector");
        assertNotNull(loraOnly.getLlmLoraConfig(),
                "loraOnly should have a non-null llmLoraConfig");
        assertNull(loraOnly.getVisionLoraConfig(),
                "loraOnly should have no visionLoraConfig");
        assertEquals(384, loraOnly.getImageResolution(),
                "Default image resolution should be 384");
        assertEquals(14, loraOnly.getPatchSize(),
                "Default patch size should be 14");
        assertEquals(576, loraOnly.getMaxImageTokens(),
                "Default maxImageTokens should be 576");
        assertEquals("vision_features", loraOnly.getVisionOutputVariable());
        assertEquals("projected_vision_features", loraOnly.getProjectorOutputVariable());

        // fullFinetune()
        VlmFineTuneConfig full = VlmFineTuneConfig.fullFinetune();
        assertFalse(full.isFreezeVisionEncoder(),
                "fullFinetune should NOT freeze the vision encoder");
        assertTrue(full.isTrainProjector(),
                "fullFinetune should train the projector");
        assertNull(full.getLlmLoraConfig(),
                "fullFinetune should have no llmLoraConfig (full weight update)");

        // projectorOnly()
        VlmFineTuneConfig projOnly = VlmFineTuneConfig.projectorOnly();
        assertTrue(projOnly.isFreezeVisionEncoder());
        assertTrue(projOnly.isTrainProjector());
        assertNull(projOnly.getLlmLoraConfig());

        // Builder with custom resolution
        VlmFineTuneConfig custom = VlmFineTuneConfig.builder()
                .imageResolution(448)
                .patchSize(14)
                .maxImageTokens(1024)
                .freezeVisionEncoder(true)
                .trainProjector(true)
                .llmLoraConfig(LoraConfig.defaultTransformer())
                .build();
        assertEquals(448, custom.getImageResolution());
        assertEquals(1024, custom.getMaxImageTokens());
    }

    // =========================================================================
    // 2. VlmFineTuneConfig validation
    // =========================================================================

    @Test
    @DisplayName("testVlmFineTuneConfigValidation - invalid imageResolution and patchSize")
    public void testVlmFineTuneConfigValidation() {
        // imageResolution not divisible by patchSize
        VlmFineTuneConfig bad = VlmFineTuneConfig.builder()
                .imageResolution(200)
                .patchSize(14)
                .maxImageTokens(100)
                .freezeVisionEncoder(true)
                .trainProjector(true)
                .build();
        assertThrows(Exception.class, bad::validate,
                "Should fail: 200 is not divisible by 14");

        // Zero patchSize
        VlmFineTuneConfig zeroPatch = VlmFineTuneConfig.builder()
                .imageResolution(224)
                .patchSize(0)
                .maxImageTokens(256)
                .freezeVisionEncoder(true)
                .trainProjector(true)
                .build();
        assertThrows(Exception.class, zeroPatch::validate,
                "Should fail: patchSize=0");

        // Zero imageResolution
        VlmFineTuneConfig zeroRes = VlmFineTuneConfig.builder()
                .imageResolution(0)
                .patchSize(14)
                .maxImageTokens(100)
                .freezeVisionEncoder(true)
                .trainProjector(true)
                .build();
        assertThrows(Exception.class, zeroRes::validate,
                "Should fail: imageResolution=0");

        // Nothing trainable
        VlmFineTuneConfig nothingTrainable = VlmFineTuneConfig.builder()
                .imageResolution(224)
                .patchSize(14)
                .maxImageTokens(256)
                .freezeVisionEncoder(true)
                .trainProjector(false)
                .llmLoraConfig(null)
                .build();
        assertThrows(Exception.class, nothingTrainable::validate,
                "Should fail: all components frozen/untrained");

        // visionLoraConfig set but encoder is frozen
        VlmFineTuneConfig frozenWithVisionLora = VlmFineTuneConfig.builder()
                .imageResolution(224)
                .patchSize(14)
                .maxImageTokens(256)
                .freezeVisionEncoder(true)  // frozen
                .trainProjector(true)
                .visionLoraConfig(LoraConfig.minimal())  // contradicts frozen
                .build();
        assertThrows(Exception.class, frozenWithVisionLora::validate,
                "Should fail: visionLoraConfig set but freezeVisionEncoder=true");

        // Valid config
        VlmFineTuneConfig valid = VlmFineTuneConfig.builder()
                .imageResolution(224)
                .patchSize(14)
                .maxImageTokens(256)
                .freezeVisionEncoder(true)
                .trainProjector(true)
                .llmLoraConfig(LoraConfig.defaultTransformer())
                .build();
        assertDoesNotThrow(valid::validate);
    }

    // =========================================================================
    // 3. VlmGRPOConfig extension
    // =========================================================================

    @Test
    @DisplayName("testVlmGRPOConfigExtension - verify extends GRPOConfig with VLM fields")
    public void testVlmGRPOConfigExtension() {
        VlmFineTuneConfig vlmConfig = VlmFineTuneConfig.loraOnly();

        VlmGRPOConfig config = VlmGRPOConfig.builder()
                .policyLogitVariable("logits")
                .vocabSize(32000)
                .groupSize(4)
                .vlmConfig(vlmConfig)
                .completionsPerImage(4)
                .imageAwareReward(true)
                .imageInputVariable("image_input")
                .build();

        // Verify it IS a GRPOConfig (inherited fields)
        assertInstanceOf(VlmGRPOConfig.class, config);
        assertEquals(4, config.getGroupSize(),
                "groupSize inherited from GRPOConfig");
        assertEquals(0.2, config.getClipEpsilon(), 1e-6,
                "clipEpsilon default from GRPOConfig");
        assertEquals("logits", config.getPolicyLogitVariable());
        assertEquals(32000, config.getVocabSize());

        // Verify VLM-specific fields
        assertEquals("VLM-GRPO", config.getMethodName());
        assertTrue(config.isImageAwareReward());
        assertEquals(4, config.getCompletionsPerImage());
        assertEquals("image_input", config.getImageInputVariable());
        assertEquals(vlmConfig, config.getVlmConfig());

        // effectiveVlmConfig returns set config
        assertEquals(vlmConfig, config.effectiveVlmConfig());

        // When vlmConfig is null, effectiveVlmConfig returns loraOnly default
        VlmGRPOConfig noVlmConfig = VlmGRPOConfig.builder()
                .policyLogitVariable("logits")
                .vocabSize(32000)
                .build();
        assertNotNull(noVlmConfig.effectiveVlmConfig());
        assertTrue(noVlmConfig.effectiveVlmConfig().isFreezeVisionEncoder());
    }

    // =========================================================================
    // 4. VlmGRPOConfig JSON round-trip
    // =========================================================================

    @Test
    @DisplayName("testVlmGRPOConfigSerialization - JSON round-trip")
    public void testVlmGRPOConfigSerialization() throws Exception {
        VlmGRPOConfig config = VlmGRPOConfig.builder()
                .policyLogitVariable("logits")
                .vocabSize(32000)
                .groupSize(4)
                .vlmConfig(VlmFineTuneConfig.loraOnly())
                .completionsPerImage(4)
                .imageAwareReward(true)
                .build();

        ObjectMapper mapper = new ObjectMapper();
        String json = mapper.writeValueAsString(config);
        assertNotNull(json);
        assertFalse(json.isEmpty());
        log.debug("VlmGRPOConfig JSON: {}", json);

        VlmGRPOConfig restored = mapper.readValue(json, VlmGRPOConfig.class);
        assertEquals(config.getGroupSize(), restored.getGroupSize());
        assertEquals(config.getVocabSize(), restored.getVocabSize());
        assertEquals(config.getPolicyLogitVariable(), restored.getPolicyLogitVariable());
        assertEquals(config.getCompletionsPerImage(), restored.getCompletionsPerImage());
        assertEquals(config.isImageAwareReward(), restored.isImageAwareReward());
        assertEquals(config.getMethodName(), restored.getMethodName());
    }

    // =========================================================================
    // 5. VisionEncodePatches op — single image
    // =========================================================================

    @Test
    @DisplayName("testVisionEncodePatchesOp - [1, 3, 224, 224] image with patchSize=14")
    public void testVisionEncodePatchesOp() {
        // 224x224 image with patchSize=14: 16 patches per side, 256 total
        // patchDim = 3 * 14 * 14 = 588
        int batch = 1, channels = 3, height = 224, width = 224, patchSize = 14;
        int numPatchesPerSide = height / patchSize;  // 16
        int numPatches = numPatchesPerSide * numPatchesPerSide;  // 256
        int patchDim = channels * patchSize * patchSize;  // 588

        INDArray image = Nd4j.linspace(DataType.FLOAT, 0.0, 1.0,
                (long) batch * channels * height * width)
                .reshape(batch, channels, height, width);

        INDArray[] outputs = Nd4j.exec(new VisionEncodePatches(image, patchSize));
        assertNotNull(outputs);
        assertEquals(2, outputs.length, "Expected 2 outputs");

        INDArray patches = outputs[0];
        INDArray numPatchesScalar = outputs[1];

        assertArrayEquals(new long[]{batch, numPatches, patchDim}, patches.shape(),
                "Output 0 shape mismatch");
        assertEquals(DataType.FLOAT, patches.dataType());

        assertEquals(numPatches, numPatchesScalar.getLong(0),
                "num_patches_per_image scalar mismatch");
        assertEquals(DataType.INT64, numPatchesScalar.dataType());

        // Verify total elements are conserved
        long totalImageElements = (long) batch * channels * height * width;
        long totalPatchElements = (long) batch * numPatches * patchDim;
        assertEquals(totalImageElements, totalPatchElements,
                "Total elements should be conserved");

        // Verify first patch corresponds to top-left corner of image
        // First pixel of channel 0 in the image
        float firstImageVal = image.getFloat(0, 0, 0, 0);
        float firstPatchVal = patches.getFloat(0, 0, 0);
        assertEquals(firstImageVal, firstPatchVal, 1e-5f,
                "First patch element should match top-left pixel of channel 0");
    }

    // =========================================================================
    // 6. VisionEncodePatches op — batched
    // =========================================================================

    @Test
    @DisplayName("testVisionEncodePatchesBatched - batch of 2 images [2, 3, 224, 224]")
    public void testVisionEncodePatchesBatched() {
        int batch = 2, channels = 3, height = 224, width = 224, patchSize = 14;
        int numPatches = (height / patchSize) * (width / patchSize);  // 256
        int patchDim = channels * patchSize * patchSize;  // 588

        INDArray image = Nd4j.rand(DataType.FLOAT, batch, channels, height, width);
        INDArray[] outputs = Nd4j.exec(new VisionEncodePatches(image, patchSize));

        INDArray patches = outputs[0];
        INDArray numPatchesScalar = outputs[1];

        assertArrayEquals(new long[]{batch, numPatches, patchDim}, patches.shape());
        assertEquals(256L, numPatchesScalar.getLong(0));

        // Each batch item should produce different patches (the images are random)
        INDArray batch0 = patches.slice(0, 0);
        INDArray batch1 = patches.slice(1, 0);
        assertFalse(batch0.equalsWithEps(batch1, 1e-5),
                "Patch tensors for different images should differ");
    }

    // =========================================================================
    // 7. VlmGRPOTrainer construction
    // =========================================================================

    @Test
    @DisplayName("testVlmGRPOTrainerConstruction - construct with minimal SameDiff models")
    public void testVlmGRPOTrainerConstruction() {
        // Build a minimal policy SameDiff with required variables
        SameDiff policy = SameDiff.create();
        SDVariable input = policy.placeHolder("input", DataType.INT64, -1, -1);
        // Minimal embedding + logits variable
        SDVariable embedding = policy.var("weight_embed",
                Nd4j.randn(DataType.FLOAT, 100, 32));
        SDVariable logits = policy.var("logits",
                Nd4j.randn(DataType.FLOAT, 1, 10, 100));

        SameDiff reference = SameDiff.create();
        reference.placeHolder("input", DataType.INT64, -1, -1);
        reference.var("logits", Nd4j.randn(DataType.FLOAT, 1, 10, 100));

        VlmGRPOConfig config = VlmGRPOConfig.builder()
                .policyLogitVariable("logits")
                .vocabSize(100)
                .groupSize(2)
                .vlmConfig(VlmFineTuneConfig.loraOnly())
                .completionsPerImage(2)
                .build();

        // Stub sampling strategy
        SamplingStrategy sampler = new SamplingStrategy() {
            @Override
            public INDArray generate(SameDiff model, INDArray prompts, int maxNewTokens, String logitVariable) {
                return Nd4j.zeros(DataType.INT64, prompts.size(0), 10);
            }
            @Override
            public INDArray generateMultiple(SameDiff model, INDArray prompts, int numCompletions, int maxNewTokens, String logitVariable) {
                return Nd4j.zeros(DataType.INT64, prompts.size(0) * numCompletions, 10);
            }
        };

        // Stub reward function
        RewardFunction reward = (prompts, completions) ->
                Nd4j.ones(DataType.FLOAT, completions.size(0));

        VlmGRPOTrainer trainer = new VlmGRPOTrainer(policy, reference, config, sampler, reward);
        assertNotNull(trainer);
        assertEquals(config, trainer.getVlmGrpoConfig());
        assertNotNull(trainer.getVlmConfig());
        assertTrue(trainer.getVlmConfig().isFreezeVisionEncoder());
    }

    // =========================================================================
    // 8. VlmFineTuneFreezeVision — isTrainable respects freeze flag
    // =========================================================================

    @Test
    @DisplayName("testVlmFineTuneFreezeVision - verify vision encoder variables are frozen")
    public void testVlmFineTuneFreezeVision() {
        SameDiff policy = SameDiff.create();
        policy.placeHolder("input", DataType.INT64, -1, -1);
        // vision encoder variables
        policy.var("vision_encoder/layer0/weight", Nd4j.randn(DataType.FLOAT, 64, 32));
        policy.var("vision_encoder/layer0/bias",   Nd4j.randn(DataType.FLOAT, 64));
        // projector variables
        policy.var("projector/weight", Nd4j.randn(DataType.FLOAT, 512, 64));
        // LLM backbone variables
        policy.var("llm/q_proj",       Nd4j.randn(DataType.FLOAT, 32, 32));
        policy.var("logits",           Nd4j.randn(DataType.FLOAT, 1, 10, 100));

        SameDiff reference = SameDiff.create();
        reference.placeHolder("input", DataType.INT64, -1, -1);
        reference.var("logits", Nd4j.randn(DataType.FLOAT, 1, 10, 100));

        VlmGRPOConfig frozenConfig = VlmGRPOConfig.builder()
                .policyLogitVariable("logits")
                .vocabSize(100)
                .groupSize(2)
                .vlmConfig(VlmFineTuneConfig.loraOnly())  // freezeVisionEncoder=true
                .completionsPerImage(2)
                .build();

        SamplingStrategy sampler = new SamplingStrategy() {
            @Override
            public INDArray generate(SameDiff model, INDArray prompts, int maxNewTokens, String logitVariable) {
                return Nd4j.zeros(DataType.INT64, prompts.size(0), 5);
            }
            @Override
            public INDArray generateMultiple(SameDiff model, INDArray prompts, int numCompletions, int maxNewTokens, String logitVariable) {
                return Nd4j.zeros(DataType.INT64, prompts.size(0) * numCompletions, 5);
            }
        };
        RewardFunction reward = (prompts, completions) ->
                Nd4j.ones(DataType.FLOAT, completions.size(0));

        VlmGRPOTrainer frozenTrainer = new VlmGRPOTrainer(
                policy, reference, frozenConfig, sampler, reward);

        // Check isTrainable for each variable type
        for (SDVariable var : policy.variables()) {
            if (var.getVariableType() != VariableType.VARIABLE) continue;
            String name = var.name();
            boolean trainable = frozenTrainer.isTrainable(var);
            if (name.startsWith(VlmGRPOTrainer.VISION_ENCODER_PREFIX)) {
                assertFalse(trainable, "Vision encoder var '" + name
                        + "' should be frozen (not trainable)");
            } else if (name.startsWith(VlmGRPOTrainer.PROJECTOR_PREFIX)
                       || name.startsWith("llm")) {
                assertTrue(trainable, "Non-vision var '" + name
                        + "' should be trainable");
            }
        }

        // Now with unfrozen vision encoder
        VlmGRPOConfig unfrozenConfig = VlmGRPOConfig.builder()
                .policyLogitVariable("logits")
                .vocabSize(100)
                .groupSize(2)
                .vlmConfig(VlmFineTuneConfig.fullFinetune())  // freezeVisionEncoder=false
                .completionsPerImage(2)
                .build();

        VlmGRPOTrainer unfrozenTrainer = new VlmGRPOTrainer(
                policy, reference, unfrozenConfig, sampler, reward);

        for (SDVariable var : policy.variables()) {
            if (var.getVariableType() != VariableType.VARIABLE) continue;
            String name = var.name();
            if (name.startsWith(VlmGRPOTrainer.VISION_ENCODER_PREFIX)) {
                assertTrue(unfrozenTrainer.isTrainable(var),
                        "Vision encoder var '" + name
                        + "' should be trainable when not frozen");
            }
        }
    }

    // =========================================================================
    // 9. VlmShareWeights config flag
    // =========================================================================

    @Test
    @DisplayName("testVlmShareWeights - verify shareWeights config flag is stored and validated")
    public void testVlmShareWeights() {
        // shareWeights=true with valid resolution/patchSize
        VlmFineTuneConfig shared = VlmFineTuneConfig.builder()
                .freezeVisionEncoder(true)
                .trainProjector(true)
                .shareWeights(true)
                .imageResolution(224)
                .patchSize(14)
                .maxImageTokens(256)
                .llmLoraConfig(LoraConfig.defaultTransformer())
                .build();

        assertTrue(shared.isShareWeights());
        assertDoesNotThrow(shared::validate);

        // shareWeights=false is the default
        VlmFineTuneConfig notShared = VlmFineTuneConfig.loraOnly();
        assertFalse(notShared.isShareWeights());

        // computeNumPatches and computePatchDim helpers
        VlmFineTuneConfig cfg = VlmFineTuneConfig.builder()
                .imageResolution(224)
                .patchSize(14)
                .maxImageTokens(256)
                .freezeVisionEncoder(true)
                .trainProjector(true)
                .build();
        assertEquals(256, cfg.computeNumPatches(),
                "(224/14)^2 = 256 patches");
        assertEquals(588, cfg.computePatchDim(3),
                "3 * 14 * 14 = 588 patch dim");

        // Verify VlmGRPOConfig propagates shareWeights through vlmConfig
        VlmGRPOConfig grpoConfig = VlmGRPOConfig.builder()
                .policyLogitVariable("logits")
                .vocabSize(32000)
                .groupSize(2)
                .vlmConfig(shared)
                .completionsPerImage(2)
                .build();
        assertTrue(grpoConfig.getVlmConfig().isShareWeights());
    }
}
