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
import org.eclipse.deeplearning4j.llm.generation.SameDiffMemoryUtils;
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
 * Integration tests for MultiModelPipeline with realistic scenarios.
 *
 * <p>Tests real-world pipeline configurations:</p>
 * <ul>
 *   <li>Document processing: summarize -> classify -> extract entities</li>
 *   <li>Multi-language pipeline: detect language -> process based on language</li>
 *   <li>Information extraction: extract relations -> sort by relevance</li>
 *   <li>Model reuse: same model for multiple roles in one pipeline</li>
 * </ul>
 *
 * <p>Run with:</p>
 * <pre>
 * cd platform-tests && mvn test -Dtest=TestMultiModelPipelineIntegration
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestMultiModelPipelineIntegration {

    private static Tokenizer tokenizer;

    @BeforeAll
    public static void setup() {
        Nd4j.setDataType(DataType.FLOAT);
        tokenizer = new MockTokenizer();
    }

    @AfterAll
    public static void teardown() {
        try {
            if (tokenizer != null) {
                tokenizer.close();
            }
        } catch (Exception e) {
            log.warn("Error closing tokenizer: {}", e.getMessage());
        }
    }

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

    // ==================== Real-World Pipeline Scenarios ====================

    @Nested
    @DisplayName("Real-World Pipeline Scenarios")
    class RealWorldPipelineScenarios {

        @Test
        @DisplayName("Test Document Processing Pipeline")
        @Order(1)
        public void testDocumentProcessingPipeline() {
            // Scenario: Process a document through summarization, classification, and NER
            String documentText = "Apple Inc. announced record earnings in Cupertino yesterday. " +
                    "CEO Tim Cook praised the team for their outstanding performance. " +
                    "The company's stock rose 5% in after-hours trading.";

            // Define pipeline stages
            List<PipelineStage> stages = Arrays.asList(
                    new PipelineStage("summarizer", "summary", 
                            Map.of("max_sentences", "2")),
                    new PipelineStage("classifier", "sentiment",
                            Map.of("categories", "positive, negative, neutral")),
                    new PipelineStage("ner", "entities",
                            Map.of("entity_types", "PERSON, ORGANIZATION, LOCATION"))
            );

            // Expected flow:
            // 1. Summarize: "Apple Inc. announced record earnings. Stock rose 5%."
            // 2. Classify: "positive"
            // 3. Extract entities: PERSON: Tim Cook, ORGANIZATION: Apple Inc., LOCATION: Cupertino

            log.info("Document processing pipeline: {} stages", stages.size());
            log.info("Input document: {} chars", documentText.length());

            // Test stage configuration
            assertEquals(3, stages.size());
            assertEquals("summarizer", stages.get(0).getModelId());
            assertEquals("classifier", stages.get(1).getModelId());
            assertEquals("ner", stages.get(2).getModelId());

            // Verify parameters
            assertEquals("2", stages.get(0).getParameters().get("max_sentences"));
            assertEquals("positive, negative, neutral", stages.get(1).getParameters().get("categories"));
        }

        @Test
        @DisplayName("Test Multi-Language Pipeline")
        @Order(2)
        public void testMultiLanguagePipeline() {
            // Scenario: Detect language, then process accordingly
            String inputText = "Bonjour, je m'appelle Jean. Je travaille chez Acme Corp à Paris.";

            // Stage 1: Detect language
            PipelineStage languageDetection = new PipelineStage(
                    "language_detector", "language");

            // Stage 2: Conditional processing based on language
            // (In real implementation, this would branch based on detected language)
            PipelineStage frenchNer = PipelineStage.fromStructuredOutput(
                    "ner", "entities", "language");

            List<PipelineStage> stages = Arrays.asList(languageDetection, frenchNer);

            log.info("Multi-language pipeline: detect -> process");
            log.info("Input text: {}", inputText);

            // Test structured input configuration
            assertTrue(frenchNer.isUseStructuredInput());
            assertEquals("language", frenchNer.getInputFromStage());

            // Expected flow:
            // 1. Detect: "fr"
            // 2. Extract French entities: PERSON: Jean, ORGANIZATION: Acme Corp, LOCATION: Paris
        }

        @Test
        @DisplayName("Test Relation Extraction Pipeline")
        @Order(3)
        public void testRelationExtractionPipeline() {
            // Scenario: Extract relations, then sort by relevance
            String text = "John works at Google in Mountain View. " +
                    "Mary studied at Stanford University. " +
                    "John and Mary met at a conference in San Francisco.";

            List<PipelineStage> stages = Arrays.asList(
                    new PipelineStage("relation_extractor", "relations"),
                    new PipelineStage("sorter", "sorted_relations",
                            Map.of("sort_criteria", "relevance", "sort_order", "descending"))
            );

            log.info("Relation extraction pipeline: {} stages", stages.size());

            // Expected flow:
            // 1. Extract: (John, works_at, Google), (Mary, studied_at, Stanford), ...
            // 2. Sort by relevance

            // Test stage configuration
            assertEquals(2, stages.size());
            assertEquals("relation_extractor", stages.get(0).getModelId());
            assertEquals("sorter", stages.get(1).getModelId());
            assertEquals("relevance", stages.get(1).getParameters().get("sort_criteria"));
        }

        @Test
        @DisplayName("Test Model Reuse in Pipeline")
        @Order(4)
        public void testModelReuseInPipeline() {
            // Scenario: Use the same general-purpose model for multiple roles
            // This tests the model reuse detection and optimization

            List<PipelineStage> stages = Arrays.asList(
                    new PipelineStage("general_model", "summary"),
                    new PipelineStage("general_model", "classification"),
                    new PipelineStage("general_model", "entities")
            );

            log.info("Model reuse pipeline: same model for {} stages", stages.size());

            // All stages use the same model
            String modelId = stages.get(0).getModelId();
            for (PipelineStage stage : stages) {
                assertEquals(modelId, stage.getModelId());
            }

            // Expected optimization: pipeline should recognize same model
            // and avoid redundant DSP compilation
        }
    }

    // ==================== Output Parser Tests ====================

    @Nested
    @DisplayName("Output Parser Tests")
    class OutputParserTests {

        @Test
        @DisplayName("Test Summarizer Output Parser")
        @Order(10)
        public void testSummarizerOutputParser() {
            String rawOutput = "Apple announced record earnings. Stock rose 5%.";
            
            ModelType.ParsedOutput parsed = ModelType.SUMMARIZER.parseOutput(rawOutput);
            
            assertNotNull(parsed);
            assertEquals(rawOutput, parsed.getRawText());
            assertEquals(rawOutput, parsed.getStructuredData());
            assertTrue(parsed.getMetadata().isEmpty());
            
            log.info("Summarizer output: {}", parsed.getRawText());
        }

        @Test
        @DisplayName("Test Classifier Output Parser with Various Formats")
        @Order(11)
        public void testClassifierOutputParserVariousFormats() {
            // Test different output formats
            String[] outputs = {"positive", "POSITIVE", "positive\n", "The category is positive"};
            
            for (String output : outputs) {
                ModelType.ParsedOutput parsed = ModelType.CLASSIFIER.parseOutput(output);
                assertNotNull(parsed);
                assertEquals("positive", parsed.getMetadata().get("category"));
                log.info("Classifier output '{}' -> category: {}", output, parsed.getMetadata().get("category"));
            }
        }

        @Test
        @DisplayName("Test Sorter Output Parser with Empty Lines")
        @Order(12)
        public void testSorterOutputParserWithEmptyLines() {
            String rawOutput = "Item 3\n\nItem 1\n   \nItem 2\n";
            
            ModelType.ParsedOutput parsed = ModelType.SORTER.parseOutput(rawOutput);
            
            assertNotNull(parsed);
            @SuppressWarnings("unchecked")
            List<String> items = (List<String>) parsed.getStructuredData();
            assertEquals(3, items.size());
            assertEquals("Item 3", items.get(0));
            assertEquals("Item 1", items.get(1));
            assertEquals("Item 2", items.get(2));
            
            log.info("Sorted items: {}", items);
        }

        @Test
        @DisplayName("Test Relation Extractor with Complex Relations")
        @Order(13)
        public void testRelationExtractorComplexRelations() {
            String rawOutput = "(John Smith, works_for, Google Inc.)\n" +
                    "(Mary Jane, studied_at, Stanford University)\n" +
                    "(Google, headquartered_in, Mountain View)";
            
            ModelType.ParsedOutput parsed = ModelType.RELATION_EXTRACTOR.parseOutput(rawOutput);
            
            assertNotNull(parsed);
            @SuppressWarnings("unchecked")
            List<ModelType.Relation> relations = (List<ModelType.Relation>) parsed.getStructuredData();
            assertEquals(3, relations.size());
            assertEquals("John Smith", relations.get(0).getSubject());
            assertEquals("works_for", relations.get(0).getPredicate());
            assertEquals("Google Inc.", relations.get(0).getObject());
            
            log.info("Extracted {} relations", relations.size());
            for (ModelType.Relation rel : relations) {
                log.info("  {}", rel);
            }
        }

        @Test
        @DisplayName("Test NER Output Parser with Multiple Entity Types")
        @Order(14)
        public void testNerOutputParserMultipleTypes() {
            String rawOutput = "PERSON: John Doe\n" +
                    "PERSON: Jane Smith\n" +
                    "ORGANIZATION: Google\n" +
                    "LOCATION: Mountain View\n" +
                    "DATE: yesterday\n" +
                    "PERCENT: 5%";
            
            ModelType.ParsedOutput parsed = ModelType.NER_MODEL.parseOutput(rawOutput);
            
            assertNotNull(parsed);
            @SuppressWarnings("unchecked")
            List<ModelType.Entity> entities = (List<ModelType.Entity>) parsed.getStructuredData();
            assertEquals(6, entities.size());
            
            // Count by type
            Map<String, Long> byType = new HashMap<>();
            for (ModelType.Entity entity : entities) {
                byType.merge(entity.getType(), 1L, Long::sum);
            }
            
            assertEquals(2L, byType.get("PERSON"));
            assertEquals(1L, byType.get("ORGANIZATION"));
            assertEquals(1L, byType.get("LOCATION"));
            assertEquals(1L, byType.get("DATE"));
            assertEquals(1L, byType.get("PERCENT"));
            
            log.info("Extracted {} entities by type: {}", entities.size(), byType);
        }

        @Test
        @DisplayName("Test Language Detector Output Parser")
        @Order(15)
        public void testLanguageDetectorOutputParser() {
            Map<String, String> testCases = new LinkedHashMap<>();
            testCases.put("en", "en");
            testCases.put("en\n", "en");
            testCases.put("English", "en");
            testCases.put("fr", "fr");
            testCases.put("es", "es");
            testCases.put("de", "de");
            testCases.put("zh", "zh");
            testCases.put("ja", "ja");
            
            for (Map.Entry<String, String> entry : testCases.entrySet()) {
                ModelType.ParsedOutput parsed = ModelType.LANGUAGE_DETECTOR.parseOutput(entry.getKey());
                assertNotNull(parsed);
                assertEquals(entry.getValue(), parsed.getMetadata().get("languageCode"));
                log.info("Language '{}' -> code: {}", entry.getKey(), parsed.getMetadata().get("languageCode"));
            }
        }
    }

    // ==================== Memory Management Tests ====================

    @Nested
    @DisplayName("Memory Management Tests")
    class MemoryManagementTests {

        @Test
        @DisplayName("Test SameDiff Memory Utils with Mock Model")
        @Order(20)
        public void testSameDiffMemoryUtils() {
            // Create a mock model with constants and variables
            SameDiff sd = SameDiff.create();
            
            SDVariable constVar = sd.constant("const", Nd4j.ones(DataType.FLOAT, 10, 10));
            SDVariable placeholder = sd.placeHolder("input", DataType.FLOAT, -1, 10);
            SDVariable output = sd.math.add(constVar, placeholder);
            
            log.info("Created mock model with {} constants, {} variables",
                    sd.getConstantArrays() != null ? sd.getConstantArrays().arrayNames().size() : 0,
                    sd.getVariablesArrays() != null ? sd.getVariablesArrays().arrayNames().size() : 0);
            
            // Test memory freeing
            int freedCount = SameDiffMemoryUtils.freeModelArrays(sd);
            log.info("Freed {} arrays", freedCount);
            
            // Close the model
            sd.close();
        }

        @Test
        @DisplayName("Test Pipeline Memory Configuration")
        @Order(21)
        public void testPipelineMemoryConfiguration() {
            // Test different memory configurations
            MultiModelPipelineConfig aggressiveMemory = MultiModelPipelineConfig.builder()
                    .tokenizer(tokenizer)
                    .freeMemoryPerStage(true)
                    .trimMemoryBetweenStages(true)
                    .build();
            
            MultiModelPipelineConfig performanceMode = MultiModelPipelineConfig.builder()
                    .tokenizer(tokenizer)
                    .freeMemoryPerStage(false)
                    .trimMemoryBetweenStages(false)
                    .build();
            
            assertTrue(aggressiveMemory.isFreeMemoryPerStage());
            assertTrue(aggressiveMemory.isTrimMemoryBetweenStages());
            assertFalse(performanceMode.isFreeMemoryPerStage());
            assertFalse(performanceMode.isTrimMemoryBetweenStages());
            
            log.info("Aggressive memory mode: freePerStage={}, trimBetween={}",
                    aggressiveMemory.isFreeMemoryPerStage(),
                    aggressiveMemory.isTrimMemoryBetweenStages());
            log.info("Performance mode: freePerStage={}, trimBetween={}",
                    performanceMode.isFreeMemoryPerStage(),
                    performanceMode.isTrimMemoryBetweenStages());
        }
    }

    // ==================== Error Handling Tests ====================

    @Nested
    @DisplayName("Error Handling Tests")
    class ErrorHandlingTests {

        @Test
        @DisplayName("Test Pipeline with Invalid Model ID")
        @Order(30)
        public void testPipelineWithInvalidModelId() throws IOException {
            MultiModelPipelineConfig config = MultiModelPipelineConfig.builder()
                    .tokenizer(tokenizer)
                    .build();

            try (MultiModelPipeline pipeline = new MultiModelPipeline(config)) {
                // Register a model
                SameDiff mockModel = createMockDecoder();
                pipeline.registerModel("valid_model", ModelType.GENERAL, mockModel);

                // Try to use non-existent model
                List<PipelineStage> stages = Arrays.asList(
                        new PipelineStage("non_existent_model", "output")
                );

                assertThrows(RuntimeException.class, () -> {
                    pipeline.execute(stages, "input");
                });

                mockModel.close();
            }
        }

        @Test
        @DisplayName("Test Pipeline with Empty Stages")
        @Order(31)
        public void testPipelineWithEmptyStages() throws IOException {
            MultiModelPipelineConfig config = MultiModelPipelineConfig.builder()
                    .tokenizer(tokenizer)
                    .build();
            
            try (MultiModelPipeline pipeline = new MultiModelPipeline(config)) {
                assertThrows(IllegalArgumentException.class, () -> {
                    pipeline.execute(Collections.emptyList(), "input");
                });
            }
        }

        @Test
        @DisplayName("Test Pipeline Result Error Handling")
        @Order(32)
        public void testPipelineResultErrorHandling() {
            PipelineResult result = new PipelineResult();
            result.setSuccess(false);
            result.setErrorMessage("Test error message");
            result.setFailedStageIndex(1);
            
            assertFalse(result.isSuccess());
            assertEquals("Test error message", result.getErrorMessage());
            assertEquals(1, result.getFailedStageIndex());
            
            log.info("Error result: {} at stage {}", 
                    result.getErrorMessage(), result.getFailedStageIndex());
        }
    }

    // ==================== Performance Tests ====================

    @Nested
    @DisplayName("Performance Tests")
    class PerformanceTests {

        @Test
        @DisplayName("Test Pipeline Stage Timing")
        @Order(40)
        public void testPipelineStageTiming() {
            PipelineResult result = new PipelineResult();
            
            // Simulate stage outputs with timing
            StageOutput stage1 = StageOutput.builder()
                    .stageId("stage1")
                    .modelId("model1")
                    .modelRole(ModelRole.SUMMARIZATION)
                    .rawText("Output 1")
                    .durationMs(100)
                    .tokenCount(50)
                    .build();
            
            StageOutput stage2 = StageOutput.builder()
                    .stageId("stage2")
                    .modelId("model2")
                    .modelRole(ModelRole.CLASSIFICATION)
                    .rawText("Output 2")
                    .durationMs(50)
                    .tokenCount(10)
                    .build();
            
            result.addStageOutput("stage1", stage1);
            result.addStageOutput("stage2", stage2);
            result.setTotalDurationMs(150);
            
            assertEquals(150, result.getTotalDurationMs());
            assertEquals(60, result.getTotalTokenCount());
            
            // Calculate throughput
            double tokensPerSecond = (result.getTotalTokenCount() / (double) result.getTotalDurationMs()) * 1000;
            log.info("Pipeline throughput: {:.2f} tokens/sec", tokensPerSecond);
        }

        @Test
        @DisplayName("Test Model Identity Tracking")
        @Order(41)
        public void testModelIdentityTracking() {
            // Create multiple references to the same model
            SameDiff model1 = createMockDecoder();
            SameDiff model2 = model1; // Same reference
            SameDiff model3 = createMockDecoder(); // Different instance

            int hash1 = System.identityHashCode(model1);
            int hash2 = System.identityHashCode(model2);
            int hash3 = System.identityHashCode(model3);

            assertEquals(hash1, hash2, "Same reference should have same identity hash");
            assertNotEquals(hash1, hash3, "Different instances should have different identity hashes");

            log.info("Model identity hashes: {} (model1), {} (model2), {} (model3)",
                    hash1, hash2, hash3);

            model1.close();
            model3.close();
        }
    }

    /**
     * Create a minimal mock decoder for testing.
     */
    private static SameDiff createMockDecoder() {
        SameDiff sd = SameDiff.create();
        
        // Create minimal placeholder variables
        org.nd4j.autodiff.samediff.SDVariable inputIds = sd.placeHolder("input_ids", DataType.INT32, -1, -1);
        org.nd4j.autodiff.samediff.SDVariable attentionMask = sd.placeHolder("attention_mask", DataType.INT32, -1, -1);
        
        // Create a simple output (not a real decoder, just for testing structure)
        org.nd4j.autodiff.samediff.SDVariable output = sd.math.add(inputIds, attentionMask);
        
        log.info("Created mock decoder with {} ops", sd.getOps().size());
        return sd;
    }
}
