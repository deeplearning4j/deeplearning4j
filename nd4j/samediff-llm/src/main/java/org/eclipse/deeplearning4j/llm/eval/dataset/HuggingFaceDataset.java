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
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.function.BiFunction;

/**
 * Downloads and loads datasets from the HuggingFace datasets-server REST API.
 * Uses the rows endpoint to fetch data in pages.
 *
 * For datasets with simple flat fields, use {@link HfFieldMapping}.
 * For datasets with nested/complex structures (e.g., ARC choices.text/choices.label),
 * use {@link #createWithTransformer} and provide a custom row parser.
 */
@Slf4j
public class HuggingFaceDataset implements EvalDataset {

    private static final String BASE_URL = "https://datasets-server.huggingface.co";
    private static final int PAGE_SIZE = 100;
    private static final int CONNECT_TIMEOUT_MS = 30000;
    private static final int READ_TIMEOUT_MS = 60000;

    private final String datasetName;
    private final List<EvalSample> samples;

    @Data
    @Builder
    public static class HfFieldMapping {
        @Builder.Default private String idField = "id";
        @Builder.Default private String inputField = "question";
        @Builder.Default private String choicesField = "choices";
        @Builder.Default private String referenceField = "answer";
        @Builder.Default private String referencesField = "answers";
        @Builder.Default private String subjectField = "subject";
        @Builder.Default private String imagePathField = "image";
    }

    private HuggingFaceDataset(String name, List<EvalSample> samples) {
        this.datasetName = name;
        this.samples = samples;
    }

    /**
     * Create a dataset by downloading from HuggingFace using field mapping.
     */
    public static HuggingFaceDataset create(String dataset, String config, String split,
                                              HfFieldMapping mapping) throws IOException {
        return create(dataset, config, split, mapping, 0);
    }

    /**
     * Create a dataset with a maximum number of samples using field mapping.
     */
    public static HuggingFaceDataset create(String dataset, String config, String split,
                                              HfFieldMapping mapping, int maxSamples) throws IOException {
        return downloadAndParse(dataset, config, split, maxSamples,
                (row, index) -> parseRow(row, mapping, index));
    }

    /**
     * Create a dataset with a custom row transformer for complex/nested schemas.
     *
     * @param dataset HF dataset name
     * @param config config/subset name (null for default)
     * @param split split name
     * @param maxSamples max samples (0 = all)
     * @param rowTransformer function that takes (JsonNode row, int index) and returns EvalSample
     */
    public static HuggingFaceDataset createWithTransformer(String dataset, String config, String split,
                                                             int maxSamples,
                                                             BiFunction<JsonNode, Integer, EvalSample> rowTransformer) throws IOException {
        return downloadAndParse(dataset, config, split, maxSamples, rowTransformer);
    }

    private static HuggingFaceDataset downloadAndParse(String dataset, String config, String split,
                                                         int maxSamples,
                                                         BiFunction<JsonNode, Integer, EvalSample> parser) throws IOException {
        log.info("Downloading HuggingFace dataset: {}/{}/{}", dataset, config, split);
        List<EvalSample> samples = new ArrayList<>();
        ObjectMapper mapper = new ObjectMapper();
        int offset = 0;

        while (true) {
            StringBuilder urlBuilder = new StringBuilder(BASE_URL)
                    .append("/rows?dataset=").append(dataset)
                    .append("&split=").append(split)
                    .append("&offset=").append(offset)
                    .append("&length=").append(PAGE_SIZE);
            if (config != null && !config.isEmpty()) {
                urlBuilder.append("&config=").append(config);
            }

            String json = fetchUrl(urlBuilder.toString());
            JsonNode root = mapper.readTree(json);
            JsonNode rows = root.get("rows");

            if (rows == null || !rows.isArray() || rows.size() == 0) break;

            for (JsonNode rowWrapper : rows) {
                JsonNode row = rowWrapper.has("row") ? rowWrapper.get("row") : rowWrapper;
                EvalSample sample = parser.apply(row, samples.size());
                if (sample != null) {
                    samples.add(sample);
                }

                if (maxSamples > 0 && samples.size() >= maxSamples) break;
            }

            if (maxSamples > 0 && samples.size() >= maxSamples) break;

            offset += rows.size();
            if (rows.size() < PAGE_SIZE) break;

            log.debug("Downloaded {} samples so far...", samples.size());
        }

        log.info("Downloaded {} samples from HuggingFace dataset '{}'", samples.size(), dataset);
        return new HuggingFaceDataset(dataset, samples);
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
        if ("test".equals(split) || "validation".equals(split)) {
            return Collections.unmodifiableList(samples);
        }
        return Collections.emptyList();
    }

    @Override
    public Iterator<EvalSample> iterator() {
        return samples.iterator();
    }

    private static EvalSample parseRow(JsonNode row, HfFieldMapping mapping, int index) {
        String id = getTextOrDefault(row, mapping.getIdField(), String.valueOf(index));
        String input = getTextOrDefault(row, mapping.getInputField(), "");

        List<String> choices = null;
        if (row.has(mapping.getChoicesField())) {
            JsonNode choicesNode = row.get(mapping.getChoicesField());
            if (choicesNode.isArray()) {
                choices = new ArrayList<>();
                for (JsonNode c : choicesNode) {
                    choices.add(c.asText());
                }
            }
        }

        List<String> references = new ArrayList<>();
        if (row.has(mapping.getReferencesField())) {
            JsonNode refsNode = row.get(mapping.getReferencesField());
            if (refsNode.isArray()) {
                for (JsonNode ref : refsNode) {
                    references.add(ref.asText());
                }
            }
        } else if (row.has(mapping.getReferenceField())) {
            JsonNode refNode = row.get(mapping.getReferenceField());
            if (refNode.isArray()) {
                for (JsonNode ref : refNode) {
                    references.add(ref.asText());
                }
            } else {
                references.add(refNode.asText());
            }
        }

        String subject = getTextOrNull(row, mapping.getSubjectField());
        String imagePath = getTextOrNull(row, mapping.getImagePathField());

        return EvalSample.builder()
                .id(id)
                .input(input)
                .choices(choices)
                .references(references)
                .subject(subject)
                .imagePath(imagePath)
                .build();
    }

    static String fetchUrl(String urlStr) throws IOException {
        URL url = new URL(urlStr);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("GET");
        conn.setConnectTimeout(CONNECT_TIMEOUT_MS);
        conn.setReadTimeout(READ_TIMEOUT_MS);
        conn.setRequestProperty("User-Agent", "DL4J-EvalFramework/1.0");

        int code = conn.getResponseCode();
        if (code == HttpURLConnection.HTTP_MOVED_TEMP || code == HttpURLConnection.HTTP_MOVED_PERM
                || code == 307 || code == 308) {
            String redirect = conn.getHeaderField("Location");
            conn.disconnect();
            return fetchUrl(redirect);
        }

        if (code != HttpURLConnection.HTTP_OK) {
            conn.disconnect();
            throw new IOException("HuggingFace API returned HTTP " + code + " for " + urlStr);
        }

        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
        }
        conn.disconnect();
        return sb.toString();
    }

    /**
     * Utility to extract a string array from a JsonNode.
     */
    public static List<String> jsonArrayToStringList(JsonNode arrayNode) {
        if (arrayNode == null || !arrayNode.isArray()) return Collections.emptyList();
        List<String> result = new ArrayList<>();
        for (JsonNode item : arrayNode) {
            result.add(item.asText());
        }
        return result;
    }

    private static String getTextOrDefault(JsonNode node, String field, String defaultValue) {
        if (node.has(field) && !node.get(field).isNull()) return node.get(field).asText();
        return defaultValue;
    }

    private static String getTextOrNull(JsonNode node, String field) {
        if (node.has(field) && !node.get(field).isNull()) return node.get(field).asText();
        return null;
    }
}
