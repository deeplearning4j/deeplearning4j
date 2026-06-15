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

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.*;

/**
 * Loads evaluation datasets from CSV or TSV files.
 */
@Slf4j
public class CsvDataset implements EvalDataset {

    private final String datasetName;
    private final List<EvalSample> samples;

    @Data
    @Builder
    public static class ColumnMapping {
        @Builder.Default private String inputColumn = "question";
        @Builder.Default private String referenceColumn = "answer";
        private String subjectColumn;
        private String idColumn;
        @Builder.Default private String delimiter = ",";
        @Builder.Default private boolean hasHeader = true;
        private List<String> choiceColumns;
    }

    public CsvDataset(String name, File csvFile) throws IOException {
        this(name, csvFile, ColumnMapping.builder().build());
    }

    public CsvDataset(String name, File csvFile, ColumnMapping mapping) throws IOException {
        this.datasetName = name;
        this.samples = loadFile(csvFile, mapping);
        log.info("Loaded {} samples from CSV dataset '{}'", samples.size(), name);
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

    private List<EvalSample> loadFile(File file, ColumnMapping mapping) throws IOException {
        List<EvalSample> result = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(Files.newInputStream(file.toPath()), StandardCharsets.UTF_8))) {

            String delimiter = mapping.getDelimiter();
            Map<String, Integer> headerMap = null;

            String firstLine = reader.readLine();
            if (firstLine == null) return result;

            if (mapping.isHasHeader()) {
                headerMap = parseHeader(firstLine, delimiter);
            } else {
                EvalSample sample = parseLine(firstLine, delimiter, null, mapping, 0);
                if (sample != null) result.add(sample);
            }

            String line;
            int lineNum = 1;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                try {
                    EvalSample sample = parseLine(line, delimiter, headerMap, mapping, lineNum);
                    if (sample != null) result.add(sample);
                } catch (Exception e) {
                    log.warn("Skipping malformed CSV line {}: {}", lineNum, e.getMessage());
                }
                lineNum++;
            }
        }
        return result;
    }

    private Map<String, Integer> parseHeader(String headerLine, String delimiter) {
        String[] headers = splitCsv(headerLine, delimiter);
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < headers.length; i++) {
            map.put(headers[i].trim(), i);
        }
        return map;
    }

    private EvalSample parseLine(String line, String delimiter, Map<String, Integer> headerMap,
                                  ColumnMapping mapping, int lineNum) {
        String[] fields = splitCsv(line, delimiter);

        String input = getField(fields, headerMap, mapping.getInputColumn());
        String reference = getField(fields, headerMap, mapping.getReferenceColumn());
        String subject = mapping.getSubjectColumn() != null ?
                getField(fields, headerMap, mapping.getSubjectColumn()) : null;
        String id = mapping.getIdColumn() != null ?
                getField(fields, headerMap, mapping.getIdColumn()) : String.valueOf(lineNum);

        List<String> choices = null;
        if (mapping.getChoiceColumns() != null && !mapping.getChoiceColumns().isEmpty()) {
            choices = new ArrayList<>();
            for (String col : mapping.getChoiceColumns()) {
                String val = getField(fields, headerMap, col);
                if (val != null) choices.add(val);
            }
        }

        List<String> references = reference != null ? List.of(reference) : Collections.emptyList();

        return EvalSample.builder()
                .id(id)
                .input(input)
                .choices(choices)
                .references(references)
                .subject(subject)
                .build();
    }

    private String getField(String[] fields, Map<String, Integer> headerMap, String column) {
        if (column == null) return null;
        int idx;
        if (headerMap != null && headerMap.containsKey(column)) {
            idx = headerMap.get(column);
        } else {
            try {
                idx = Integer.parseInt(column);
            } catch (NumberFormatException e) {
                return null;
            }
        }
        if (idx >= 0 && idx < fields.length) {
            String val = fields[idx].trim();
            if (val.length() >= 2 && val.startsWith("\"") && val.endsWith("\"")) {
                val = val.substring(1, val.length() - 1).replace("\"\"", "\"");
            }
            return val;
        }
        return null;
    }

    private static String[] splitCsv(String line, String delimiter) {
        if (",".equals(delimiter)) {
            return splitRespectingQuotes(line);
        }
        return line.split(delimiter, -1);
    }

    private static String[] splitRespectingQuotes(String line) {
        List<String> fields = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"') {
                if (inQuotes && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                    current.append('"');
                    i++;
                } else {
                    inQuotes = !inQuotes;
                    current.append(c);
                }
            } else if (c == ',' && !inQuotes) {
                fields.add(current.toString());
                current = new StringBuilder();
            } else {
                current.append(c);
            }
        }
        fields.add(current.toString());
        return fields.toArray(new String[0]);
    }
}
