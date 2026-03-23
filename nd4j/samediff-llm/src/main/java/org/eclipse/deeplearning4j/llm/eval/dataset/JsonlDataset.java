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

package org.eclipse.deeplearning4j.llm.eval.dataset;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.*;

/**
 * Loads evaluation datasets from JSONL (JSON Lines) files.
 * Each line is a JSON object representing one sample.
 * Field mappings are configurable to support different JSONL schemas.
 */
@Slf4j
public class JsonlDataset implements EvalDataset {

    private final String datasetName;
    private final List<EvalSample> samples;

    @Data
    @Builder
    public static class FieldMapping {
        @Builder.Default private String idField = "id";
        @Builder.Default private String inputField = "question";
        @Builder.Default private String choicesField = "choices";
        @Builder.Default private String referenceField = "answer";
        @Builder.Default private String referencesField = "answers";
        @Builder.Default private String subjectField = "subject";
        @Builder.Default private String imagePathField = "image";
    }

    public JsonlDataset(String name, File jsonlFile) throws IOException {
        this(name, jsonlFile, FieldMapping.builder().build());
    }

    public JsonlDataset(String name, File jsonlFile, FieldMapping mapping) throws IOException {
        this.datasetName = name;
        this.samples = loadFile(jsonlFile, mapping);
        log.info("Loaded {} samples from JSONL dataset '{}'", samples.size(), name);
    }

    public JsonlDataset(String name, InputStream inputStream, FieldMapping mapping) throws IOException {
        this.datasetName = name;
        this.samples = loadStream(inputStream, mapping);
        log.info("Loaded {} samples from JSONL dataset '{}'", samples.size(), name);
    }

    @Override
    public String name() {
        return datasetName;
    }

    @Override
    public int size() {
        return samples.size();
    }

    @Override
    public EvalSample get(int index) {
        return samples.get(index);
    }

    @Override
    public List<EvalSample> getSplit(String split) {
        if ("test".equals(split)) return Collections.unmodifiableList(samples);
        return Collections.emptyList();
    }

    @Override
    public Iterator<EvalSample> iterator() {
        return samples.iterator();
    }

    private List<EvalSample> loadFile(File file, FieldMapping mapping) throws IOException {
        try (InputStream is = Files.newInputStream(file.toPath())) {
            return loadStream(is, mapping);
        }
    }

    private List<EvalSample> loadStream(InputStream is, FieldMapping mapping) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        List<EvalSample> result = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            String line;
            int lineNum = 0;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                try {
                    JsonNode node = mapper.readTree(line);
                    EvalSample sample = parseSample(node, mapping, lineNum);
                    result.add(sample);
                } catch (Exception e) {
                    log.warn("Skipping malformed JSONL line {}: {}", lineNum, e.getMessage());
                }
                lineNum++;
            }
        }
        return result;
    }

    private EvalSample parseSample(JsonNode node, FieldMapping mapping, int lineNum) {
        String id = getTextOrDefault(node, mapping.getIdField(), String.valueOf(lineNum));
        String input = getTextOrDefault(node, mapping.getInputField(), "");

        List<String> choices = null;
        if (node.has(mapping.getChoicesField()) && node.get(mapping.getChoicesField()).isArray()) {
            choices = new ArrayList<>();
            for (JsonNode choice : node.get(mapping.getChoicesField())) {
                choices.add(choice.asText());
            }
        }

        List<String> references = new ArrayList<>();
        if (node.has(mapping.getReferencesField()) && node.get(mapping.getReferencesField()).isArray()) {
            for (JsonNode ref : node.get(mapping.getReferencesField())) {
                references.add(ref.asText());
            }
        } else if (node.has(mapping.getReferenceField())) {
            JsonNode refNode = node.get(mapping.getReferenceField());
            if (refNode.isArray()) {
                for (JsonNode ref : refNode) {
                    references.add(ref.asText());
                }
            } else {
                references.add(refNode.asText());
            }
        }

        String subject = getTextOrNull(node, mapping.getSubjectField());
        String imagePath = getTextOrNull(node, mapping.getImagePathField());

        return EvalSample.builder()
                .id(id)
                .input(input)
                .choices(choices)
                .references(references)
                .subject(subject)
                .imagePath(imagePath)
                .build();
    }

    private static String getTextOrDefault(JsonNode node, String field, String defaultValue) {
        if (node.has(field) && !node.get(field).isNull()) {
            return node.get(field).asText();
        }
        return defaultValue;
    }

    private static String getTextOrNull(JsonNode node, String field) {
        if (node.has(field) && !node.get(field).isNull()) {
            return node.get(field).asText();
        }
        return null;
    }
}
