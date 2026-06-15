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

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Defines a model type with built-in prompt templates for specific tasks.
 *
 * <p>Each model type encapsulates:
 * <ul>
 *   <li>A default prompt template for the task</li>
 *   <li>Input/output format specifications</li>
 *   <li>Post-processing logic for structured outputs</li>
 * </ul>
 * </p>
 *
 * <p>Built-in model types:</p>
 * <ul>
 *   <li>{@link #SUMMARIZER} - Text summarization with configurable length</li>
 *   <li>{@link #CLASSIFIER} - Text classification with category labels</li>
 *   <li>{@link #SORTER} - Item sorting/ranking by relevance</li>
 *   <li>{@link #RELATION_EXTRACTOR} - Entity relation extraction</li>
 *   <li>{@link #LANGUAGE_DETECTOR} - Language identification</li>
 *   <li>{@link #NER_MODEL} - Named entity recognition</li>
 * </ul>
 *
 * <p>Custom model types can be created using {@link #builder()}.</p>
 */
@Slf4j
@Getter
public class ModelType {

    /**
     * Built-in summarizer model type.
     * Produces concise summaries of long text.
     */
    public static final ModelType SUMMARIZER = builder()
            .role(ModelRole.SUMMARIZATION)
            .name("summarizer")
            .defaultPromptTemplate("Summarize the following text concisely. Capture the key points in {{max_sentences}} sentences or less:\n\n{{input}}")
            .outputParser(SummarizerOutputParser.INSTANCE)
            .defaultParams(Map.of("max_sentences", "3"))
            .build();

    /**
     * Built-in classifier model type.
     * Categorizes text into predefined classes.
     */
    public static final ModelType CLASSIFIER = builder()
            .role(ModelRole.CLASSIFICATION)
            .name("classifier")
            .defaultPromptTemplate("Classify the following text into one of these categories: {{categories}}.\n\nText: {{input}}\n\nCategory:")
            .outputParser(ClassifierOutputParser.INSTANCE)
            .defaultParams(Map.of("categories", "positive, negative, neutral"))
            .build();

    /**
     * Built-in sorter model type.
     * Orders items by relevance, priority, or other criteria.
     */
    public static final ModelType SORTER = builder()
            .role(ModelRole.SORTING)
            .name("sorter")
            .defaultPromptTemplate("Sort the following items by {{sort_criteria}} in {{sort_order}} order. Return only the sorted list, one item per line:\n\n{{input}}")
            .outputParser(SorterOutputParser.INSTANCE)
            .defaultParams(Map.of("sort_criteria", "relevance", "sort_order", "descending"))
            .build();

    /**
     * Built-in relation extractor model type.
     * Extracts relationships between entities.
     */
    public static final ModelType RELATION_EXTRACTOR = builder()
            .role(ModelRole.RELATION_EXTRACTION)
            .name("relation_extractor")
            .defaultPromptTemplate("Extract all relationships from the following text. For each relationship, identify the subject, predicate, and object. Format each as: (subject, predicate, object)\n\nText: {{input}}\n\nRelationships:")
            .outputParser(RelationExtractorOutputParser.INSTANCE)
            .build();

    /**
     * Built-in language detector model type.
     * Identifies the language of text.
     */
    public static final ModelType LANGUAGE_DETECTOR = builder()
            .role(ModelRole.LANGUAGE_CLASSIFICATION)
            .name("language_detector")
            .defaultPromptTemplate("Identify the language of the following text. Return only the ISO 639-1 language code (e.g., 'en', 'es', 'fr'):\n\n{{input}}")
            .outputParser(LanguageDetectorOutputParser.INSTANCE)
            .build();

    /**
     * Built-in named entity recognizer model type.
     * Identifies and classifies named entities.
     */
    public static final ModelType NER_MODEL = builder()
            .role(ModelRole.NAMED_ENTITY_RECOGNITION)
            .name("ner_model")
            .defaultPromptTemplate("Identify all named entities in the following text. For each entity, specify the entity type (PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT) and the entity text. Format as: ENTITY_TYPE: entity_text\n\nText: {{input}}\n\nEntities:")
            .outputParser(NerOutputParser.INSTANCE)
            .build();

    /**
     * General-purpose model type for custom prompts.
     */
    public static final ModelType GENERAL = builder()
            .role(ModelRole.GENERAL)
            .name("general")
            .defaultPromptTemplate("{{input}}")
            .outputParser(GeneralOutputParser.INSTANCE)
            .build();

    private final ModelRole role;
    private final String name;
    private final String defaultPromptTemplate;
    private final OutputParser outputParser;
    private final Map<String, String> defaultParams;

    private ModelType(Builder builder) {
        this.role = builder.role;
        this.name = builder.name;
        this.defaultPromptTemplate = builder.defaultPromptTemplate;
        this.outputParser = builder.outputParser != null ? builder.outputParser : GeneralOutputParser.INSTANCE;
        this.defaultParams = builder.defaultParams != null ? new HashMap<>(builder.defaultParams) : new HashMap<>();
    }

    /**
     * Build a prompt from the template with the given input and parameters.
     *
     * @param input the input text
     * @param params additional parameters (overrides defaults)
     * @return the rendered prompt
     */
    public String buildPrompt(String input, Map<String, String> params) {
        Map<String, String> mergedParams = new HashMap<>(defaultParams);
        if (params != null) {
            mergedParams.putAll(params);
        }
        mergedParams.put("input", input);

        String prompt = defaultPromptTemplate;
        for (Map.Entry<String, String> entry : mergedParams.entrySet()) {
            prompt = prompt.replace("{{" + entry.getKey() + "}}", entry.getValue());
        }
        return prompt;
    }

    /**
     * Parse the model output according to the role's expected format.
     *
     * @param rawOutput the raw model output
     * @return the parsed output
     */
    public ParsedOutput parseOutput(String rawOutput) {
        return outputParser.parse(rawOutput);
    }

    /**
     * Create a builder for custom model types.
     *
     * @return a new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for ModelType.
     */
    public static class Builder {
        private ModelRole role;
        private String name;
        private String defaultPromptTemplate;
        private OutputParser outputParser;
        private Map<String, String> defaultParams;

        public Builder role(ModelRole role) {
            this.role = role;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder defaultPromptTemplate(String defaultPromptTemplate) {
            this.defaultPromptTemplate = defaultPromptTemplate;
            return this;
        }

        public Builder outputParser(OutputParser outputParser) {
            this.outputParser = outputParser;
            return this;
        }

        public Builder defaultParams(Map<String, String> defaultParams) {
            this.defaultParams = defaultParams;
            return this;
        }

        public ModelType build() {
            return new ModelType(this);
        }
    }

    /**
     * Interface for parsing model outputs.
     */
    public interface OutputParser {
        ParsedOutput parse(String rawOutput);
    }

    /**
     * Represents parsed model output with structured data.
     */
    @Getter
    public static class ParsedOutput {
        private final String rawText;
        private final Object structuredData;
        private final Map<String, Object> metadata;

        private ParsedOutput(String rawText, Object structuredData, Map<String, Object> metadata) {
            this.rawText = rawText;
            this.structuredData = structuredData;
            this.metadata = metadata != null ? metadata : new HashMap<>();
        }

        public static ParsedOutput of(String rawText) {
            return new ParsedOutput(rawText, rawText, new HashMap<>());
        }

        public static ParsedOutput of(String rawText, Object structuredData) {
            return new ParsedOutput(rawText, structuredData, new HashMap<>());
        }

        public static Builder builder() {
            return new Builder();
        }

        public static class Builder {
            private String rawText;
            private Object structuredData;
            private Map<String, Object> metadata;

            public Builder rawText(String rawText) {
                this.rawText = rawText;
                return this;
            }

            public Builder structuredData(Object structuredData) {
                this.structuredData = structuredData;
                return this;
            }

            public Builder metadata(Map<String, Object> metadata) {
                this.metadata = metadata;
                return this;
            }

            public ParsedOutput build() {
                return new ParsedOutput(rawText, structuredData, metadata);
            }
        }
    }

    // ---- Built-in Output Parsers ----

    private static class SummarizerOutputParser implements OutputParser {
        static final SummarizerOutputParser INSTANCE = new SummarizerOutputParser();

        @Override
        public ParsedOutput parse(String rawOutput) {
            return ParsedOutput.of(rawOutput.trim());
        }
    }

    private static class ClassifierOutputParser implements OutputParser {
        static final ClassifierOutputParser INSTANCE = new ClassifierOutputParser();

        // Common classification categories to look for
        private static final Set<String> COMMON_CATEGORIES;
        static {
            Set<String> cats = new HashSet<>();
            cats.add("positive"); cats.add("negative"); cats.add("neutral");
            cats.add("spam"); cats.add("ham");
            cats.add("yes"); cats.add("no");
            cats.add("true"); cats.add("false");
            cats.add("happy"); cats.add("sad"); cats.add("angry");
            cats.add("good"); cats.add("bad"); cats.add("ok");
            cats.add("high"); cats.add("medium"); cats.add("low");
            cats.add("urgent"); cats.add("normal");
            COMMON_CATEGORIES = Collections.unmodifiableSet(cats);
        }

        @Override
        public ParsedOutput parse(String rawOutput) {
            String cleaned = rawOutput.trim().toLowerCase();
            String category;
            
            // First, try to find common category keywords anywhere in the output
            for (String cat : COMMON_CATEGORIES) {
                if (cleaned.contains(cat)) {
                    category = cat;
                    Map<String, Object> metadata = new HashMap<>();
                    metadata.put("category", category);
                    metadata.put("confidence", 1.0); // Placeholder - actual confidence from model
                    return ParsedOutput.builder()
                            .rawText(rawOutput.trim())
                            .structuredData(category)
                            .metadata(metadata)
                            .build();
                }
            }
            
            // Fallback: extract the first token
            category = cleaned.split("[\\s,;]+")[0];
            Map<String, Object> metadata = new HashMap<>();
            metadata.put("category", category);
            metadata.put("confidence", 1.0);
            return ParsedOutput.builder()
                    .rawText(rawOutput.trim())
                    .structuredData(category)
                    .metadata(metadata)
                    .build();
        }
    }

    private static class SorterOutputParser implements OutputParser {
        static final SorterOutputParser INSTANCE = new SorterOutputParser();

        @Override
        public ParsedOutput parse(String rawOutput) {
            String[] lines = rawOutput.trim().split("\\n");
            List<String> sortedItems = new ArrayList<>();
            for (String line : lines) {
                String trimmed = line.trim();
                if (!trimmed.isEmpty()) {
                    sortedItems.add(trimmed);
                }
            }
            Map<String, Object> metadata = new HashMap<>();
            metadata.put("itemCount", sortedItems.size());
            return ParsedOutput.builder()
                    .rawText(rawOutput.trim())
                    .structuredData(sortedItems)
                    .metadata(metadata)
                    .build();
        }
    }

    private static class RelationExtractorOutputParser implements OutputParser {
        static final RelationExtractorOutputParser INSTANCE = new RelationExtractorOutputParser();

        @Override
        public ParsedOutput parse(String rawOutput) {
            List<Relation> relations = new ArrayList<>();
            // Parse (subject, predicate, object) tuples
            String[] lines = rawOutput.trim().split("\\n");
            for (String line : lines) {
                line = line.trim();
                if (line.startsWith("(") && line.endsWith(")")) {
                    String content = line.substring(1, line.length() - 1);
                    String[] parts = content.split(",");
                    if (parts.length >= 3) {
                        relations.add(new Relation(
                                parts[0].trim(),
                                parts[1].trim(),
                                parts[2].trim()
                        ));
                    }
                }
            }
            Map<String, Object> metadata = new HashMap<>();
            metadata.put("relationCount", relations.size());
            return ParsedOutput.builder()
                    .rawText(rawOutput.trim())
                    .structuredData(relations)
                    .metadata(metadata)
                    .build();
        }
    }

    @Getter
    public static class Relation {
        private final String subject;
        private final String predicate;
        private final String object;

        public Relation(String subject, String predicate, String object) {
            this.subject = subject;
            this.predicate = predicate;
            this.object = object;
        }

        @Override
        public String toString() {
            return "(" + subject + ", " + predicate + ", " + object + ")";
        }
    }

    private static class LanguageDetectorOutputParser implements OutputParser {
        static final LanguageDetectorOutputParser INSTANCE = new LanguageDetectorOutputParser();

        @Override
        public ParsedOutput parse(String rawOutput) {
            String languageCode = rawOutput.trim().toLowerCase();
            // Extract ISO 639-1 code (2 letters)
            if (languageCode.length() > 2) {
                languageCode = languageCode.substring(0, 2);
            }
            Map<String, Object> metadata = new HashMap<>();
            metadata.put("languageCode", languageCode);
            return ParsedOutput.builder()
                    .rawText(rawOutput.trim())
                    .structuredData(languageCode)
                    .metadata(metadata)
                    .build();
        }
    }

    private static class NerOutputParser implements OutputParser {
        static final NerOutputParser INSTANCE = new NerOutputParser();

        @Override
        public ParsedOutput parse(String rawOutput) {
            List<Entity> entities = new ArrayList<>();
            String[] lines = rawOutput.trim().split("\\n");
            for (String line : lines) {
                line = line.trim();
                if (line.contains(":")) {
                    String[] parts = line.split(":", 2);
                    if (parts.length == 2) {
                        entities.add(new Entity(
                                parts[0].trim().toUpperCase(),
                                parts[1].trim()
                        ));
                    }
                }
            }
            Map<String, Object> metadata = new HashMap<>();
            metadata.put("entityCount", entities.size());
            return ParsedOutput.builder()
                    .rawText(rawOutput.trim())
                    .structuredData(entities)
                    .metadata(metadata)
                    .build();
        }
    }

    @Getter
    public static class Entity {
        private final String type;
        private final String text;

        public Entity(String type, String text) {
            this.type = type;
            this.text = text;
        }

        @Override
        public String toString() {
            return type + ": " + text;
        }
    }

    private static class GeneralOutputParser implements OutputParser {
        static final GeneralOutputParser INSTANCE = new GeneralOutputParser();

        @Override
        public ParsedOutput parse(String rawOutput) {
            return ParsedOutput.of(rawOutput);
        }
    }
}
