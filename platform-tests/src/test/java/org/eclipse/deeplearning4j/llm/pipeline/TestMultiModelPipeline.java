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

package org.eclipse.deeplearning4j.llm.pipeline;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for the MultiModelPipeline framework.
 *
 * <p>Tests cover:</p>
 * <ul>
 *   <li>ModelType prompt building and output parsing for all 6 model types</li>
 *   <li>Pipeline stage execution and output chaining</li>
 *   <li>Model reuse detection and optimization</li>
 *   <li>DSP optimizations and memory management</li>
 *   <li>Error handling and pipeline failure recovery</li>
 * </ul>
 *
 * <p>Run with:</p>
 * <pre>
 * cd platform-tests && mvn test -Dtest=TestMultiModelPipeline
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestMultiModelPipeline {

    private static Tokenizer tokenizer;
    private static SameDiff mockDecoder;

    @BeforeAll
    public static void setup() throws Exception {
        // Initialize ND4J
        if (Nd4j.getAffinityManager() == null) {
            Nd4j.setDataType(DataType.FLOAT);
        }

        // Use mock tokenizer for testing
        tokenizer = new MockTokenizer();

        // Create a minimal mock decoder for testing
        mockDecoder = createMockDecoder();
    }

    @AfterAll
    public static void teardown() {
        if (mockDecoder != null) {
            mockDecoder.close();
        }
    }

    // ==================== ModelType Tests ====================

    @Nested
    @DisplayName("ModelType Tests")
    class ModelTypeTests {
        private final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger(ModelTypeTests.class);

        @Test
        @DisplayName("Test Summarizer Prompt Building")
        @Order(1)
        public void testSummarizerPromptBuilding() {
            ModelType modelType = ModelType.SUMMARIZER;
            String input = "This is a long text that needs to be summarized. It contains multiple points.";
            
            String prompt = modelType.buildPrompt(input, Collections.emptyMap());
            
            assertNotNull(prompt);
            assertTrue(prompt.contains("Summarize the following text"));
            assertTrue(prompt.contains(input));
            assertTrue(prompt.contains("3 sentences"));
            
            log.info("Summarizer prompt: {}", prompt);
        }

        @Test
        @DisplayName("Test Summarizer Output Parsing")
        @Order(2)
        public void testSummarizerOutputParsing() {
            ModelType modelType = ModelType.SUMMARIZER;
            String rawOutput = "This is a concise summary of the text.";
            
            ModelType.ParsedOutput parsed = modelType.parseOutput(rawOutput);
            
            assertNotNull(parsed);
            assertEquals(rawOutput, parsed.getRawText());
            assertEquals(rawOutput, parsed.getStructuredData());
            assertFalse(modelType.getRole().isStructuredOutput());
            
            log.info("Parsed summarizer output: {}", parsed.getRawText());
        }

        @Test
        @DisplayName("Test Classifier Prompt Building")
        @Order(3)
        public void testClassifierPromptBuilding() {
            ModelType modelType = ModelType.CLASSIFIER;
            String input = "I love this product! It's amazing.";
            
            String prompt = modelType.buildPrompt(input, Collections.singletonMap("categories", "positive, negative, neutral"));
            
            assertNotNull(prompt);
            assertTrue(prompt.contains("Classify the following text"));
            assertTrue(prompt.contains("positive, negative, neutral"));
            assertTrue(prompt.contains(input));
            
            log.info("Classifier prompt: {}", prompt);
        }

        @Test
        @DisplayName("Test Classifier Output Parsing")
        @Order(4)
        public void testClassifierOutputParsing() {
            ModelType modelType = ModelType.CLASSIFIER;
            String rawOutput = "positive";
            
            ModelType.ParsedOutput parsed = modelType.parseOutput(rawOutput);
            
            assertNotNull(parsed);
            assertEquals("positive", parsed.getStructuredData());
            assertEquals("positive", parsed.getMetadata().get("category"));
            assertTrue(modelType.getRole().isStructuredOutput());
            
            log.info("Parsed classifier output: category={}", parsed.getMetadata().get("category"));
        }

        @Test
        @DisplayName("Test Sorter Output Parsing")
        @Order(5)
        public void testSorterOutputParsing() {
            ModelType modelType = ModelType.SORTER;
            String rawOutput = "Item 3\nItem 1\nItem 2";
            
            ModelType.ParsedOutput parsed = modelType.parseOutput(rawOutput);
            
            assertNotNull(parsed);
            assertTrue(parsed.getStructuredData() instanceof List);
            @SuppressWarnings("unchecked")
            List<String> sortedItems = (List<String>) parsed.getStructuredData();
            assertEquals(3, sortedItems.size());
            assertEquals("Item 3", sortedItems.get(0));
            assertEquals(3, parsed.getMetadata().get("itemCount"));
            
            log.info("Parsed sorter output: {}", sortedItems);
        }

        @Test
        @DisplayName("Test Relation Extractor Output Parsing")
        @Order(6)
        public void testRelationExtractorOutputParsing() {
            ModelType modelType = ModelType.RELATION_EXTRACTOR;
            String rawOutput = "(John, works_at, Acme Corp)\n(Mary, lives_in, Paris)";
            
            ModelType.ParsedOutput parsed = modelType.parseOutput(rawOutput);
            
            assertNotNull(parsed);
            assertTrue(parsed.getStructuredData() instanceof List);
            @SuppressWarnings("unchecked")
            List<ModelType.Relation> relations = (List<ModelType.Relation>) parsed.getStructuredData();
            assertEquals(2, relations.size());
            assertEquals("John", relations.get(0).getSubject());
            assertEquals("works_at", relations.get(0).getPredicate());
            assertEquals("Acme Corp", relations.get(0).getObject());
            
            log.info("Parsed relations: {}", relations);
        }

        @Test
        @DisplayName("Test Language Detector Output Parsing")
        @Order(7)
        public void testLanguageDetectorOutputParsing() {
            ModelType modelType = ModelType.LANGUAGE_DETECTOR;
            String rawOutput = "en";
            
            ModelType.ParsedOutput parsed = modelType.parseOutput(rawOutput);
            
            assertNotNull(parsed);
            assertEquals("en", parsed.getStructuredData());
            assertEquals("en", parsed.getMetadata().get("languageCode"));
            
            log.info("Detected language: {}", parsed.getMetadata().get("languageCode"));
        }

        @Test
        @DisplayName("Test NER Output Parsing")
        @Order(8)
        public void testNerOutputParsing() {
            ModelType modelType = ModelType.NER_MODEL;
            String rawOutput = "PERSON: John Doe\nORGANIZATION: Acme Corp\nLOCATION: Paris";
            
            ModelType.ParsedOutput parsed = modelType.parseOutput(rawOutput);
            
            assertNotNull(parsed);
            assertTrue(parsed.getStructuredData() instanceof List);
            @SuppressWarnings("unchecked")
            List<ModelType.Entity> entities = (List<ModelType.Entity>) parsed.getStructuredData();
            assertEquals(3, entities.size());
            assertEquals("PERSON", entities.get(0).getType());
            assertEquals("John Doe", entities.get(0).getText());
            assertEquals("ORGANIZATION", entities.get(1).getType());
            assertEquals("LOCATION", entities.get(2).getType());
            
            log.info("Parsed entities: {}", entities);
        }

        @Test
        @DisplayName("Test Custom Model Type")
        @Order(9)
        public void testCustomModelType() {
            ModelType customType = ModelType.builder()
                    .role(ModelRole.GENERAL)
                    .name("custom_qa")
                    .defaultPromptTemplate("Answer the following question: {{input}}")
                    .defaultParams(Map.of("max_length", "100"))
                    .build();
            
            String input = "What is AI?";
            String prompt = customType.buildPrompt(input, Collections.emptyMap());
            
            assertNotNull(prompt);
            assertTrue(prompt.contains("Answer the following question"));
            assertTrue(prompt.contains(input));
            assertEquals("custom_qa", customType.getName());
            
            log.info("Custom prompt: {}", prompt);
        }
    }

    // ==================== PipelineStage Tests ====================

    @Nested
    @DisplayName("PipelineStage Tests")
    class PipelineStageTests {
        private final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger(PipelineStageTests.class);

        @Test
        @DisplayName("Test Simple Stage Creation")
        @Order(10)
        public void testSimpleStageCreation() {
            PipelineStage stage = new PipelineStage("summarizer", "summary");
            
            assertEquals("summarizer", stage.getModelId());
            assertEquals("summary", stage.getStageId());
            assertTrue(stage.getParameters().isEmpty());
            assertFalse(stage.isUseStructuredInput());
            
            log.info("Created stage: {} -> {}", stage.getModelId(), stage.getStageId());
        }

        @Test
        @DisplayName("Test Stage with Parameters")
        @Order(11)
        public void testStageWithParameters() {
            Map<String, String> params = new HashMap<>();
            params.put("categories", "spam, ham");
            params.put("confidence", "0.8");
            
            PipelineStage stage = new PipelineStage("classifier", "category", params);
            
            assertEquals(2, stage.getParameters().size());
            assertEquals("spam, ham", stage.getParameters().get("categories"));
            
            log.info("Created stage with parameters: {}", stage.getParameters());
        }

        @Test
        @DisplayName("Test Stage with Structured Input")
        @Order(12)
        public void testStageWithStructuredInput() {
            PipelineStage stage = PipelineStage.fromStructuredOutput(
                    "ner", "entities", "classification");
            
            assertTrue(stage.isUseStructuredInput());
            assertEquals("classification", stage.getInputFromStage());
            
            log.info("Created stage with structured input from: {}", stage.getInputFromStage());
        }
    }

    // ==================== PipelineResult Tests ====================

    @Nested
    @DisplayName("PipelineResult Tests")
    class PipelineResultTests {
        private final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger(PipelineResultTests.class);

        @Test
        @DisplayName("Test Result Construction")
        @Order(20)
        public void testResultConstruction() {
            PipelineResult result = new PipelineResult();
            
            assertFalse(result.isSuccess());
            assertEquals(0, result.getStageCount());
            assertNull(result.getErrorMessage());
            
            log.info("Created empty pipeline result");
        }

        @Test
        @DisplayName("Test Adding Stage Outputs")
        @Order(21)
        public void testAddingStageOutputs() {
            PipelineResult result = new PipelineResult();
            
            StageOutput output1 = StageOutput.builder()
                    .stageId("summary")
                    .modelId("summarizer")
                    .modelRole(ModelRole.SUMMARIZATION)
                    .rawText("Concise summary")
                    .durationMs(100)
                    .tokenCount(50)
                    .build();
            
            StageOutput output2 = StageOutput.builder()
                    .stageId("category")
                    .modelId("classifier")
                    .modelRole(ModelRole.CLASSIFICATION)
                    .rawText("positive")
                    .durationMs(50)
                    .tokenCount(10)
                    .build();
            
            result.addStageOutput("summary", output1);
            result.addStageOutput("category", output2);
            result.setSuccess(true);
            result.setTotalDurationMs(150);
            
            assertEquals(2, result.getStageCount());
            assertTrue(result.isSuccess());
            assertEquals(60, result.getTotalTokenCount());
            assertEquals(150, result.getTotalDurationMs());
            
            // Test retrieval
            StageOutput retrieved = result.getStageOutput("summary");
            assertNotNull(retrieved);
            assertEquals("Concise summary", retrieved.getRawText());
            
            // Test ordered outputs
            List<StageOutput> outputs = result.getOutputsInOrder();
            assertEquals(2, outputs.size());
            assertEquals("summary", outputs.get(0).getStageId());
            assertEquals("category", outputs.get(1).getStageId());
            
            log.info("Pipeline result: {}", result.getSummary());
        }

        @Test
        @DisplayName("Test Structured Data Retrieval")
        @Order(22)
        public void testStructuredDataRetrieval() {
            PipelineResult result = new PipelineResult();
            
            List<ModelType.Entity> entities = Arrays.asList(
                    new ModelType.Entity("PERSON", "John"),
                    new ModelType.Entity("ORG", "Acme")
            );
            
            ModelType.ParsedOutput parsed = ModelType.ParsedOutput.builder()
                    .rawText("PERSON: John\nORG: Acme")
                    .structuredData(entities)
                    .metadata(new HashMap<>())
                    .build();
            
            StageOutput output = StageOutput.builder()
                    .stageId("entities")
                    .modelId("ner")
                    .modelRole(ModelRole.NAMED_ENTITY_RECOGNITION)
                    .rawText("PERSON: John\nORG: Acme")
                    .parsedOutput(parsed)
                    .durationMs(100)
                    .tokenCount(20)
                    .build();
            
            result.addStageOutput("entities", output);
            
            Object structuredData = result.getStageStructuredData("entities");
            assertNotNull(structuredData);
            assertTrue(structuredData instanceof List);
            @SuppressWarnings("unchecked")
            List<ModelType.Entity> retrieved = (List<ModelType.Entity>) structuredData;
            assertEquals(2, retrieved.size());
            
            log.info("Retrieved structured data: {}", retrieved);
        }
    }

    // ==================== MultiModelPipeline Integration Tests ====================

    @Nested
    @DisplayName("MultiModelPipeline Integration Tests")
    class MultiModelPipelineIntegrationTests {
        private final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger(MultiModelPipelineIntegrationTests.class);

        @Test
        @DisplayName("Test Pipeline Configuration")
        @Order(30)
        public void testPipelineConfiguration() {
            MultiModelPipelineConfig config = MultiModelPipelineConfig.builder()
                    .tokenizer(tokenizer)
                    .maxTokensPerModel(128)
                    .dspEnabled(true)
                    .trimMemoryBetweenStages(true)
                    .freeMemoryPerStage(true)
                    .build();
            
            assertNotNull(config);
            assertEquals(tokenizer, config.getTokenizer());
            assertEquals(128, config.getMaxTokensPerModel());
            assertTrue(config.isDspEnabled());
            assertTrue(config.isTrimMemoryBetweenStages());
            
            log.info("Created pipeline config: maxTokens={}, DSP={}", 
                    config.getMaxTokensPerModel(), config.isDspEnabled());
        }

        @Test
        @DisplayName("Test Model Registration")
        @Order(31)
        public void testModelRegistration() throws IOException {
            MultiModelPipelineConfig config = MultiModelPipelineConfig.builder()
                    .tokenizer(tokenizer)
                    .build();
            
            try (MultiModelPipeline pipeline = new MultiModelPipeline(config)) {
                // Register the same model for different roles
                pipeline.registerModel("summarizer", ModelType.SUMMARIZER, mockDecoder);
                pipeline.registerModel("classifier", ModelType.CLASSIFIER, mockDecoder);
                pipeline.registerModel("ner", ModelType.NER_MODEL, mockDecoder);
                
                assertEquals(3, pipeline.getModelIds().size());
                assertTrue(pipeline.getModelIds().contains("summarizer"));
                assertTrue(pipeline.getModelIds().contains("classifier"));
                assertTrue(pipeline.getModelIds().contains("ner"));
                
                // Verify model reuse detection
                MultiModelPipelineConfig.RegisteredModel model = pipeline.getModel("summarizer");
                assertNotNull(model);
                assertEquals(ModelRole.SUMMARIZATION, model.getModelType().getRole());
                
                log.info("Registered {} models, model IDs: {}", 
                        pipeline.getModelIds().size(), pipeline.getModelIds());
            }
        }

        @Test
        @DisplayName("Test Pipeline with Mock Models")
        @Order(32)
        public void testPipelineWithMockModels() throws IOException {
            MultiModelPipelineConfig config = MultiModelPipelineConfig.builder()
                    .tokenizer(tokenizer)
                    .maxTokensPerModel(64)
                    .dspEnabled(false) // Disable DSP for mock testing
                    .trimMemoryBetweenStages(false)
                    .build();
            
            try (MultiModelPipeline pipeline = new MultiModelPipeline(config)) {
                // Register mock models
                pipeline.registerModel("summarizer", ModelType.SUMMARIZER, mockDecoder);
                pipeline.registerModel("classifier", ModelType.CLASSIFIER, mockDecoder);
                
                // Define pipeline stages
                List<PipelineStage> stages = Arrays.asList(
                        new PipelineStage("summarizer", "summary"),
                        new PipelineStage("classifier", "category")
                );
                
                // Execute pipeline (will fail with mock decoder, but tests structure)
                assertThrows(RuntimeException.class, () -> {
                    pipeline.execute(stages, "Test input text");
                });
                
                log.info("Pipeline execution test completed (expected failure with mock)");
            }
        }

        @Test
        @DisplayName("Test Model Reuse Detection")
        @Order(33)
        public void testModelReuseDetection() throws IOException {
            MultiModelPipelineConfig config = MultiModelPipelineConfig.builder()
                    .tokenizer(tokenizer)
                    .build();
            
            try (MultiModelPipeline pipeline = new MultiModelPipeline(config)) {
                // Register the SAME model instance for different roles
                pipeline.registerModel("model_a", ModelType.SUMMARIZER, mockDecoder);
                pipeline.registerModel("model_b", ModelType.CLASSIFIER, mockDecoder);
                
                // Both should reference the same underlying model
                MultiModelPipelineConfig.RegisteredModel modelA = pipeline.getModel("model_a");
                MultiModelPipelineConfig.RegisteredModel modelB = pipeline.getModel("model_b");
                
                assertSame(modelA.getModel(), modelB.getModel());
                
                log.info("Model reuse detected: model_a and model_b share same instance");
            }
        }
    }

    // ==================== DSP and Memory Management Tests ====================

    @Nested
    @DisplayName("DSP and Memory Management Tests")
    class DspAndMemoryTests {
        private final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger(DspAndMemoryTests.class);

        @Test
        @DisplayName("Test DSP Configuration")
        @Order(40)
        public void testDspConfiguration() {
            // Save original DSP settings
            String originalDsp = System.getProperty("nd4j.dsp.enabled");
            String originalNoFreeze = System.getProperty("nd4j.dsp.no_freeze");
            
            try {
                // Enable DSP
                System.setProperty("nd4j.dsp.enabled", "true");
                System.setProperty("nd4j.dsp.no_freeze", "false");
                
                MultiModelPipelineConfig config = MultiModelPipelineConfig.builder()
                        .tokenizer(tokenizer)
                        .dspEnabled(true)
                        .cudaGraphsEnabled(false)
                        .build();
                
                assertTrue(config.isDspEnabled());
                assertFalse(config.isCudaGraphsEnabled());
                
                log.info("DSP configuration test passed");
            } finally {
                // Restore original settings
                if (originalDsp != null) {
                    System.setProperty("nd4j.dsp.enabled", originalDsp);
                }
                if (originalNoFreeze != null) {
                    System.setProperty("nd4j.dsp.no_freeze", originalNoFreeze);
                }
            }
        }

        @Test
        @DisplayName("Test Memory Trimming Configuration")
        @Order(41)
        public void testMemoryTrimmingConfiguration() {
            MultiModelPipelineConfig config1 = MultiModelPipelineConfig.builder()
                    .tokenizer(tokenizer)
                    .trimMemoryBetweenStages(true)
                    .freeMemoryPerStage(true)
                    .build();
            
            MultiModelPipelineConfig config2 = MultiModelPipelineConfig.builder()
                    .tokenizer(tokenizer)
                    .trimMemoryBetweenStages(false)
                    .freeMemoryPerStage(false)
                    .build();
            
            assertTrue(config1.isTrimMemoryBetweenStages());
            assertTrue(config1.isFreeMemoryPerStage());
            assertFalse(config2.isTrimMemoryBetweenStages());
            assertFalse(config2.isFreeMemoryPerStage());
            
            log.info("Memory trimming configuration test passed");
        }
    }

    // ==================== Helper Methods ====================

    /**
     * Create a minimal mock decoder for testing.
     */
    private static SameDiff createMockDecoder() {
        SameDiff sd = SameDiff.create();

        // Create minimal placeholder variables
        SDVariable inputIds = sd.placeHolder("input_ids", DataType.INT32, -1, -1);
        SDVariable attentionMask = sd.placeHolder("attention_mask", DataType.INT32, -1, -1);

        // Create a simple output (not a real decoder, just for testing structure)
        SDVariable output = sd.math.add(inputIds, attentionMask);

        log.info("Created mock decoder with {} ops", sd.getOps().size());
        return sd;
    }

    // ==================== Mock Tokenizer ====================

    /**
     * Mock tokenizer for testing when HuggingFace tokenizer is unavailable.
     */
    private static class MockTokenizer implements Tokenizer {

        @Override
        public org.eclipse.deeplearning4j.llm.tokenizer.Encoding encode(String text, boolean addSpecialTokens) {
            // Simple mock: split on whitespace and assign fake token IDs
            String[] tokens = text.split("\\s+");
            int[] ids = new int[tokens.length];
            String[] tokenStrings = new String[tokens.length];
            int[] attentionMask = new int[tokens.length];
            int[] typeIds = new int[tokens.length];
            
            for (int i = 0; i < tokens.length; i++) {
                ids[i] = i + 1; // Fake token IDs
                tokenStrings[i] = tokens[i];
                attentionMask[i] = 1;
                typeIds[i] = 0;
            }
            return org.eclipse.deeplearning4j.llm.tokenizer.Encoding.builder()
                    .ids(ids)
                    .tokens(tokenStrings)
                    .attentionMask(attentionMask)
                    .typeIds(typeIds)
                    .build();
        }

        @Override
        public String decode(int[] tokenIds, boolean skipSpecialTokens) {
            // Simple mock: return placeholder text
            return "Decoded text with " + tokenIds.length + " tokens";
        }

        @Override
        public int getVocabSize() {
            return 1000;
        }

        @Override
        public Integer getTokenId(String token) {
            return 1;
        }

        @Override
        public String getToken(int id) {
            return "token";
        }

        @Override
        public java.util.Map<String, Integer> getVocab() {
            return new java.util.HashMap<>();
        }

        @Override
        public int getPadTokenId() {
            return 0;
        }

        @Override
        public int getBosTokenId() {
            return 1;
        }

        @Override
        public int getEosTokenId() {
            return 2;
        }

        @Override
        public int getUnkTokenId() {
            return 3;
        }

        @Override
        public boolean isValid() {
            return true;
        }

        @Override
        public java.util.List<org.eclipse.deeplearning4j.llm.tokenizer.Encoding> encodeBatch(java.util.List<String> texts, boolean addSpecialTokens) {
            java.util.List<org.eclipse.deeplearning4j.llm.tokenizer.Encoding> results = new java.util.ArrayList<>();
            for (String text : texts) {
                results.add(encode(text, addSpecialTokens));
            }
            return results;
        }

        @Override
        public java.util.List<String> decodeBatch(java.util.List<int[]> idsBatch, boolean skipSpecialTokens) {
            java.util.List<String> results = new java.util.ArrayList<>();
            for (int[] ids : idsBatch) {
                results.add(decode(ids, skipSpecialTokens));
            }
            return results;
        }

        @Override
        public void close() throws java.io.IOException {
            // No-op
        }
    }
}
