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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff.pipeline;

import com.google.gson.JsonObject;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.pipeline.*;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for samediff-pipeline-core module classes.
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class PipelineCoreTest {

    @TempDir
    Path tempDir;

    // ==================== ModelFormat Tests ====================

    @Test
    public void testModelFormatFromExtension() {
        assertEquals(ModelFormat.SAFETENSORS, ModelFormat.fromExtension("safetensors"));
        assertEquals(ModelFormat.GGUF, ModelFormat.fromExtension("gguf"));
        assertEquals(ModelFormat.ONNX, ModelFormat.fromExtension("onnx"));
        assertEquals(ModelFormat.PYTORCH, ModelFormat.fromExtension("pt"));
        assertEquals(ModelFormat.PYTORCH, ModelFormat.fromExtension("bin"));
        assertEquals(ModelFormat.SDZ, ModelFormat.fromExtension("sdz"));
        assertEquals(ModelFormat.UNKNOWN, ModelFormat.fromExtension("xyz"));
    }

    @Test
    public void testModelFormatFromFilename() {
        assertEquals(ModelFormat.SAFETENSORS, ModelFormat.fromFilename("model.safetensors"));
        assertEquals(ModelFormat.GGUF, ModelFormat.fromFilename("llama-7b.gguf"));
        assertEquals(ModelFormat.ONNX, ModelFormat.fromFilename("model.onnx"));
        assertEquals(ModelFormat.SDZ, ModelFormat.fromFilename("model.sdz"));
        assertEquals(ModelFormat.UNKNOWN, ModelFormat.fromFilename("model.txt"));
    }

    // ==================== LoadConfig Tests ====================

    @Test
    public void testLoadConfigDefaults() {
        PipelineLoader.LoadConfig config = PipelineLoader.LoadConfig.defaults();

        assertEquals("float32", config.getDataType());
        assertEquals("cpu", config.getDevice());
        assertTrue(config.useMmap());
        assertTrue(config.cacheConvertedModel());
        assertTrue(config.convertToFloat32());
        assertTrue(config.dequantize());
        assertNotNull(config.getCacheDirectory());
    }

    @Test
    public void testLoadConfigBuilder() {
        PipelineLoader.LoadConfig config = PipelineLoader.LoadConfig.builder()
                .dataType("float16")
                .device("cuda:0")
                .useMmap(false)
                .cacheConvertedModel(false)
                .convertToFloat32(false)
                .dequantize(false)
                .build();

        assertEquals("float16", config.getDataType());
        assertEquals("cuda:0", config.getDevice());
        assertFalse(config.useMmap());
        assertFalse(config.cacheConvertedModel());
        assertFalse(config.convertToFloat32());
        assertFalse(config.dequantize());
    }

    @Test
    public void testLoadConfigBuilderWithCacheDirectory() {
        File customCacheDir = tempDir.resolve("custom-cache").toFile();
        PipelineLoader.LoadConfig config = PipelineLoader.LoadConfig.builder()
                .cacheDirectory(customCacheDir)
                .build();

        assertEquals(customCacheDir, config.getCacheDirectory());
    }

    // ==================== GenerationConfig Tests ====================

    @Test
    public void testGenerationConfigFromJson() {
        JsonObject json = new JsonObject();
        json.addProperty("temperature", 0.7);
        json.addProperty("top_p", 0.9);
        json.addProperty("top_k", 50);
        json.addProperty("max_length", 2048);
        json.addProperty("max_new_tokens", 512);
        json.addProperty("do_sample", true);
        json.addProperty("bos_token_id", 1);
        json.addProperty("eos_token_id", 2);
        json.addProperty("pad_token_id", 0);

        GenerationConfig config = GenerationConfig.fromJson(json);

        assertEquals(0.7, config.getTemperature(), 0.001);
        assertEquals(0.9, config.getTopP(), 0.001);
        assertEquals(50, config.getTopK());
        assertEquals(2048, config.getMaxLength());
        assertEquals(512, config.getMaxNewTokens());
        assertTrue(config.getDoSample());
        assertEquals(1, config.getBosTokenId());
        assertEquals(2, config.getEosTokenId());
        assertEquals(0, config.getPadTokenId());
    }

    @Test
    public void testGenerationConfigEffectiveValues() {
        GenerationConfig config = GenerationConfig.builder().build();

        // Test defaults
        assertEquals(1.0, config.getEffectiveTemperature(), 0.001);
        assertEquals(1.0, config.getEffectiveTopP(), 0.001);
        assertEquals(50, config.getEffectiveTopK());
    }

    @Test
    public void testGenerationConfigFromFile() throws IOException {
        File configFile = tempDir.resolve("generation_config.json").toFile();
        try (FileWriter writer = new FileWriter(configFile)) {
            writer.write("{\"temperature\": 0.8, \"max_new_tokens\": 256}");
        }

        GenerationConfig config = GenerationConfig.fromFile(configFile);
        assertEquals(0.8, config.getTemperature(), 0.001);
        assertEquals(256, config.getMaxNewTokens());
    }

    @Test
    public void testGenerationConfigLoadIfExists() throws IOException {
        File dir = tempDir.toFile();
        File configFile = new File(dir, "generation_config.json");
        try (FileWriter writer = new FileWriter(configFile)) {
            writer.write("{\"temperature\": 0.5}");
        }

        Optional<GenerationConfig> config = GenerationConfig.loadIfExists(dir);
        assertTrue(config.isPresent());
        assertEquals(0.5, config.get().getTemperature(), 0.001);

        // Test non-existent
        File emptyDir = tempDir.resolve("empty").toFile();
        emptyDir.mkdir();
        Optional<GenerationConfig> empty = GenerationConfig.loadIfExists(emptyDir);
        assertFalse(empty.isPresent());
    }

    // ==================== TokenizerConfig Tests ====================

    @Test
    public void testTokenizerConfigFromJson() {
        JsonObject json = new JsonObject();
        json.addProperty("tokenizer_class", "LlamaTokenizer");
        json.addProperty("bos_token", "<s>");
        json.addProperty("eos_token", "</s>");
        json.addProperty("pad_token", "<pad>");
        json.addProperty("unk_token", "<unk>");
        json.addProperty("model_max_length", 4096);
        json.addProperty("chat_template", "{% for message in messages %}{{ message['content'] }}{% endfor %}");

        TokenizerConfig config = TokenizerConfig.fromJson(json);

        assertEquals("LlamaTokenizer", config.getTokenizerClass());
        assertEquals("<s>", config.getBosToken());
        assertEquals("</s>", config.getEosToken());
        assertEquals("<pad>", config.getPadToken());
        assertEquals("<unk>", config.getUnkToken());
        assertEquals(4096, config.getModelMaxLength());
        assertTrue(config.hasChatTemplate());
    }

    @Test
    public void testTokenizerConfigFromFile() throws IOException {
        File configFile = tempDir.resolve("tokenizer_config.json").toFile();
        try (FileWriter writer = new FileWriter(configFile)) {
            writer.write("{\"tokenizer_class\": \"GPT2Tokenizer\", \"model_max_length\": 1024}");
        }

        TokenizerConfig config = TokenizerConfig.fromFile(configFile);
        assertEquals("GPT2Tokenizer", config.getTokenizerClass());
        assertEquals(1024, config.getModelMaxLength());
    }

    // ==================== SpecialTokensMap Tests ====================

    @Test
    public void testSpecialTokensMapFromJson() {
        JsonObject json = new JsonObject();
        json.addProperty("bos_token", "<s>");
        json.addProperty("eos_token", "</s>");
        json.addProperty("pad_token", "<pad>");
        json.addProperty("unk_token", "<unk>");

        SpecialTokensMap map = SpecialTokensMap.fromJson(json);

        assertEquals("<s>", map.getBosTokenString());
        assertEquals("</s>", map.getEosTokenString());
        assertEquals("<pad>", map.getPadTokenString());
        assertEquals("<unk>", map.getUnkTokenString());
    }

    @Test
    public void testSpecialTokensMapWithTokenInfo() {
        JsonObject json = new JsonObject();

        JsonObject bosToken = new JsonObject();
        bosToken.addProperty("content", "<|begin|>");
        bosToken.addProperty("lstrip", false);
        bosToken.addProperty("rstrip", false);
        bosToken.addProperty("single_word", false);
        bosToken.addProperty("special", true);
        json.add("bos_token", bosToken);

        SpecialTokensMap map = SpecialTokensMap.fromJson(json);
        assertNotNull(map.getBosToken());
        assertEquals("<|begin|>", map.getBosTokenString());
    }

    // ==================== PreprocessorConfig Tests ====================

    @Test
    public void testPreprocessorConfigFromJson() {
        JsonObject json = new JsonObject();
        json.addProperty("image_processor_type", "CLIPImageProcessor");
        json.addProperty("image_size", 224);
        json.addProperty("do_normalize", true);
        json.addProperty("do_resize", true);
        json.addProperty("do_center_crop", true);
        json.addProperty("rescale_factor", 0.00392156862745098);

        com.google.gson.JsonArray imageMean = new com.google.gson.JsonArray();
        imageMean.add(0.48145466);
        imageMean.add(0.4578275);
        imageMean.add(0.40821073);
        json.add("image_mean", imageMean);

        com.google.gson.JsonArray imageStd = new com.google.gson.JsonArray();
        imageStd.add(0.26862954);
        imageStd.add(0.26130258);
        imageStd.add(0.27577711);
        json.add("image_std", imageStd);

        PreprocessorConfig config = PreprocessorConfig.fromJson(json);

        assertEquals("CLIPImageProcessor", config.getImageProcessorType());
        assertEquals(224, config.getImageSize());
        assertTrue(config.getDoNormalize());
        assertTrue(config.getDoResize());
        assertTrue(config.getDoCenterCrop());
        assertEquals(224, config.getEffectiveSize());

        double[] mean = config.getImageMeanArray();
        assertEquals(3, mean.length);
        assertEquals(0.48145466, mean[0], 0.0001);

        double[] std = config.getImageStdArray();
        assertEquals(3, std.length);
        assertEquals(0.26862954, std[0], 0.0001);
    }

    @Test
    public void testPreprocessorConfigDefaults() {
        PreprocessorConfig config = PreprocessorConfig.builder().build();

        // Default ImageNet values
        double[] mean = config.getImageMeanArray();
        assertEquals(3, mean.length);
        assertEquals(0.485, mean[0], 0.001);

        double[] std = config.getImageStdArray();
        assertEquals(3, std.length);
        assertEquals(0.229, std[0], 0.001);

        assertEquals(224, config.getEffectiveSize());
        assertEquals(1.0 / 255.0, config.getEffectiveRescaleFactor(), 0.0001);
    }

    // ==================== SchedulerConfig Tests ====================

    @Test
    public void testSchedulerConfigFromJson() {
        JsonObject json = new JsonObject();
        json.addProperty("_class_name", "EulerDiscreteScheduler");
        json.addProperty("num_train_timesteps", 1000);
        json.addProperty("beta_start", 0.0001);
        json.addProperty("beta_end", 0.02);
        json.addProperty("beta_schedule", "scaled_linear");
        json.addProperty("prediction_type", "epsilon");
        json.addProperty("use_karras_sigmas", true);

        SchedulerConfig config = SchedulerConfig.fromJson(json);

        assertEquals("EulerDiscreteScheduler", config.getSchedulerClassName());
        assertEquals(1000, config.getEffectiveNumTrainTimesteps());
        assertEquals(0.0001, config.getEffectiveBetaStart(), 0.00001);
        assertEquals(0.02, config.getEffectiveBetaEnd(), 0.0001);
        assertEquals("scaled_linear", config.getBetaSchedule());
        assertEquals("epsilon", config.getEffectivePredictionType());
        assertTrue(config.isEulerScheduler());
        assertFalse(config.isDDPMScheduler());
    }

    @Test
    public void testSchedulerConfigDefaults() {
        SchedulerConfig config = SchedulerConfig.builder().build();

        assertEquals("DDPMScheduler", config.getSchedulerClassName());
        assertEquals(1000, config.getEffectiveNumTrainTimesteps());
        assertEquals(0.0001, config.getEffectiveBetaStart(), 0.00001);
        assertEquals(0.02, config.getEffectiveBetaEnd(), 0.0001);
        assertEquals("linear", config.getEffectiveBetaSchedule());
        assertEquals("epsilon", config.getEffectivePredictionType());
    }

    // ==================== ChatTemplate Tests ====================

    @Test
    public void testChatTemplateSimpleFormat() {
        ChatTemplate template = ChatTemplate.builder()
                .addGenerationPrompt(true)
                .build();

        List<ChatTemplate.Message> messages = Arrays.asList(
                ChatTemplate.Message.system("You are a helpful assistant."),
                ChatTemplate.Message.user("Hello!"),
                ChatTemplate.Message.assistant("Hi there!")
        );

        String result = template.apply(messages);
        assertNotNull(result);
        assertTrue(result.contains("You are a helpful assistant."));
        assertTrue(result.contains("Hello!"));
        assertTrue(result.contains("Hi there!"));
    }

    @Test
    public void testChatTemplateChatML() {
        ChatTemplate template = ChatTemplate.chatMl();

        List<ChatTemplate.Message> messages = Arrays.asList(
                ChatTemplate.Message.user("What is 2+2?")
        );

        String result = template.apply(messages);
        assertNotNull(result);
        // ChatML format should include the im_start/im_end tokens
        assertTrue(result.contains("<|im_start|>") || result.contains("user"));
    }

    @Test
    public void testChatTemplateLlama() {
        ChatTemplate template = ChatTemplate.llama();

        List<ChatTemplate.Message> messages = Arrays.asList(
                ChatTemplate.Message.user("Hello")
        );

        String result = template.apply(messages);
        assertNotNull(result);
    }

    @Test
    public void testChatTemplateFromTokenizerConfig() {
        TokenizerConfig tokenizerConfig = TokenizerConfig.builder()
                .bosToken("<s>")
                .eosToken("</s>")
                .chatTemplate("{% for message in messages %}{{ message['content'] }}{% endfor %}")
                .build();

        ChatTemplate template = ChatTemplate.fromTokenizerConfig(tokenizerConfig);
        assertEquals("<s>", template.getBosToken());
        assertEquals("</s>", template.getEosToken());
        assertNotNull(template.getTemplate());
    }

    // ==================== ModelManifest Tests ====================

    @Test
    public void testModelManifestFromDirectorySafetensors() throws IOException {
        File modelDir = tempDir.resolve("safetensors-model").toFile();
        modelDir.mkdirs();

        // Create a dummy safetensors file
        new File(modelDir, "model.safetensors").createNewFile();

        // Create config.json
        try (FileWriter writer = new FileWriter(new File(modelDir, "config.json"))) {
            writer.write("{\"model_type\": \"llama\", \"hidden_size\": 4096}");
        }

        ModelManifest manifest = ModelManifest.fromDirectory(modelDir);

        assertEquals(ModelFormat.SAFETENSORS, manifest.getFormat());
        assertEquals(ModelManifest.ModelType.SINGLE, manifest.getType());
        assertFalse(manifest.isPipeline());
        assertFalse(manifest.isSharded());
        assertEquals("llama", manifest.getArchitecture());
        assertEquals(4096, manifest.getHiddenSize());
        assertNotNull(manifest.getWeightFiles());
        assertEquals(1, manifest.getWeightFiles().size());
    }

    @Test
    public void testModelManifestFromDirectoryWithConfigs() throws IOException {
        File modelDir = tempDir.resolve("full-model").toFile();
        modelDir.mkdirs();

        // Create model file
        new File(modelDir, "model.safetensors").createNewFile();

        // Create config.json
        try (FileWriter writer = new FileWriter(new File(modelDir, "config.json"))) {
            writer.write("{\"model_type\": \"llama\", \"vocab_size\": 32000, \"num_attention_heads\": 32}");
        }

        // Create generation_config.json
        try (FileWriter writer = new FileWriter(new File(modelDir, "generation_config.json"))) {
            writer.write("{\"max_new_tokens\": 512, \"temperature\": 0.7}");
        }

        // Create tokenizer_config.json
        try (FileWriter writer = new FileWriter(new File(modelDir, "tokenizer_config.json"))) {
            writer.write("{\"tokenizer_class\": \"LlamaTokenizer\", \"bos_token\": \"<s>\", \"eos_token\": \"</s>\"}");
        }

        ModelManifest manifest = ModelManifest.fromDirectory(modelDir);

        assertTrue(manifest.hasGenerationConfig());
        assertTrue(manifest.hasTokenizerConfig());
        assertEquals(32000, manifest.getVocabSize());
        assertEquals(32, manifest.getNumAttentionHeads());
        assertTrue(manifest.isLikelyLLM());
    }

    @Test
    public void testModelManifestIsLikelyVisionModel() throws IOException {
        File modelDir = tempDir.resolve("vision-model").toFile();
        modelDir.mkdirs();

        new File(modelDir, "model.safetensors").createNewFile();

        // Create config.json for vision model
        try (FileWriter writer = new FileWriter(new File(modelDir, "config.json"))) {
            writer.write("{\"model_type\": \"vit\"}");
        }

        // Create preprocessor_config.json
        try (FileWriter writer = new FileWriter(new File(modelDir, "preprocessor_config.json"))) {
            writer.write("{\"image_size\": 224, \"do_normalize\": true}");
        }

        ModelManifest manifest = ModelManifest.fromDirectory(modelDir);

        assertTrue(manifest.hasPreprocessorConfig());
        assertTrue(manifest.isLikelyVisionModel());
        assertFalse(manifest.isLikelyLLM());
    }

    @Test
    public void testModelManifestNonExistentDirectory() {
        File nonExistent = new File("/non/existent/path");
        assertThrows(IOException.class, () -> ModelManifest.fromDirectory(nonExistent));
    }

    // ==================== WeightMapIndex Tests ====================

    @Test
    public void testWeightMapIndexFromJson() {
        JsonObject json = new JsonObject();

        JsonObject metadata = new JsonObject();
        metadata.addProperty("total_size", 14000000000L);
        json.add("metadata", metadata);

        JsonObject weightMap = new JsonObject();
        weightMap.addProperty("model.embed_tokens.weight", "model-00001-of-00002.safetensors");
        weightMap.addProperty("model.layers.0.self_attn.q_proj.weight", "model-00001-of-00002.safetensors");
        weightMap.addProperty("model.layers.31.mlp.up_proj.weight", "model-00002-of-00002.safetensors");
        weightMap.addProperty("lm_head.weight", "model-00002-of-00002.safetensors");
        json.add("weight_map", weightMap);

        WeightMapIndex index = WeightMapIndex.fromJson(json);

        assertEquals(14000000000L, index.getTotalSize());
        Set<String> shards = index.getShardFiles();
        assertEquals(2, shards.size());
        assertTrue(shards.contains("model-00001-of-00002.safetensors"));
        assertTrue(shards.contains("model-00002-of-00002.safetensors"));

        assertEquals("model-00001-of-00002.safetensors",
                index.getShardForWeight("model.embed_tokens.weight"));
    }

    @Test
    public void testWeightMapIndexFromFile() throws IOException {
        File indexFile = tempDir.resolve("model.safetensors.index.json").toFile();
        try (FileWriter writer = new FileWriter(indexFile)) {
            writer.write("{\n" +
                    "  \"metadata\": {\"total_size\": 5000000000},\n" +
                    "  \"weight_map\": {\n" +
                    "    \"layer1.weight\": \"shard1.safetensors\",\n" +
                    "    \"layer2.weight\": \"shard2.safetensors\"\n" +
                    "  }\n" +
                    "}");
        }

        WeightMapIndex index = WeightMapIndex.fromFile(indexFile);

        assertEquals(5000000000L, index.getTotalSize());
        assertEquals(2, index.getShardFiles().size());
    }

    // ==================== ModelIndex Tests ====================

    @Test
    public void testModelIndexFromJson() {
        JsonObject json = new JsonObject();
        json.addProperty("_class_name", "StableDiffusionPipeline");
        json.addProperty("_diffusers_version", "0.25.0");

        JsonObject textEncoder = new JsonObject();
        textEncoder.addProperty("class_name", "CLIPTextModel");
        json.add("text_encoder", textEncoder);

        JsonObject unet = new JsonObject();
        unet.addProperty("class_name", "UNet2DConditionModel");
        json.add("unet", unet);

        JsonObject vae = new JsonObject();
        vae.addProperty("class_name", "AutoencoderKL");
        json.add("vae", vae);

        ModelIndex index = ModelIndex.fromJson(json);

        assertEquals("StableDiffusionPipeline", index.getClassName());
        assertTrue(index.isPipeline());

        Set<String> components = index.getComponentNames();
        assertEquals(3, components.size());
        assertTrue(components.contains("text_encoder"));
        assertTrue(components.contains("unet"));
        assertTrue(components.contains("vae"));
    }

    @Test
    public void testModelIndexFromFile() throws IOException {
        File indexFile = tempDir.resolve("model_index.json").toFile();
        try (FileWriter writer = new FileWriter(indexFile)) {
            writer.write("{\n" +
                    "  \"_class_name\": \"DiffusionPipeline\",\n" +
                    "  \"scheduler\": {\"class_name\": \"DDPMScheduler\"},\n" +
                    "  \"unet\": {\"class_name\": \"UNet2DModel\"}\n" +
                    "}");
        }

        ModelIndex index = ModelIndex.fromFile(indexFile);

        assertEquals("DiffusionPipeline", index.getClassName());
        assertTrue(index.isPipeline());
        assertEquals(2, index.getComponentNames().size());
    }
}
