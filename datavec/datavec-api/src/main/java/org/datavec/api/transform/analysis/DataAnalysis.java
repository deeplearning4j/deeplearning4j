/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.transform.analysis;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.analysis.columns.ColumnAnalysis;
import org.datavec.api.transform.metadata.CategoricalMetaData;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.serde.JsonSerializer;
import org.datavec.api.transform.serde.YamlSerializer;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.node.ArrayNode;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;

/**
 * The DataAnalysis class represents analysis (summary statistics) for a data set.
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Data
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public class DataAnalysis implements Serializable {
    private static final String COL_NAME = "columnName";
    private static final String COL_IDX = "columnIndex";
    private static final String COL_TYPE = "columnType";
    private static final String CATEGORICAL_STATE_NAMES = "stateNames";
    private static final String ANALYSIS = "analysis";
    private static final String DATA_ANALYSIS = "DataAnalysis";

    private Schema schema;
    private List<ColumnAnalysis> columnAnalysis;

    protected DataAnalysis(){
        //No arg for JSON
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        int nCol = schema.numColumns();

        int maxNameLength = 0;
        for (String s : schema.getColumnNames()) {
            maxNameLength = Math.max(maxNameLength, s.length());
        }

        //Header:
        sb.append(String.format("%-6s", "idx")).append(String.format("%-" + (maxNameLength + 8) + "s", "name"))
                        .append(String.format("%-15s", "type")).append("analysis").append("\n");

        for (int i = 0; i < nCol; i++) {
            String colName = schema.getName(i);
            ColumnType type = schema.getType(i);
            ColumnAnalysis analysis = columnAnalysis.get(i);
            String paddedName = String.format("%-" + (maxNameLength + 8) + "s", "\"" + colName + "\"");
            sb.append(String.format("%-6d", i)).append(paddedName).append(String.format("%-15s", type)).append(analysis)
                            .append("\n");
        }

        return sb.toString();
    }

    public ColumnAnalysis getColumnAnalysis(String column) {
        return columnAnalysis.get(schema.getIndexOfColumn(column));
    }

    /**
     * Convert the DataAnalysis object to JSON format
     */
    public String toJson() {
        try{
            return new JsonSerializer().getObjectMapper().writeValueAsString(this);
        } catch (IOException e){
            throw new RuntimeException(e);
        }
    }

    /**
     * Convert the DataAnalysis object to YAML format
     */
    public String toYaml() {
        try{
            return new YamlSerializer().getObjectMapper().writeValueAsString(this);
        } catch (IOException e){
            throw new RuntimeException(e);
        }
    }

    /**
     * Deserialize a JSON DataAnalysis String that was previously serialized with {@link #toJson()}
     */
    public static DataAnalysis fromJson(String json) {
        try{
            return new JsonSerializer().getObjectMapper().readValue(json, DataAnalysis.class);
        } catch (Exception e){
            //Legacy format
            ObjectMapper om = new JsonSerializer().getObjectMapper();
            return fromMapper(om, json);
        }
    }

    /**
     * Deserialize a YAML DataAnalysis String that was previously serialized with {@link #toYaml()}
     */
    public static DataAnalysis fromYaml(String yaml) {
        try{
            return new YamlSerializer().getObjectMapper().readValue(yaml, DataAnalysis.class);
        } catch (Exception e){
            //Legacy format
            ObjectMapper om = new YamlSerializer().getObjectMapper();
            return fromMapper(om, yaml);
        }
    }

    private static DataAnalysis fromMapper(ObjectMapper om, String json) {

        List<ColumnMetaData> meta = new ArrayList<>();
        List<ColumnAnalysis> analysis = new ArrayList<>();
        try {
            JsonNode node = om.readTree(json);
            Iterator<String> fieldNames = node.fieldNames();
            boolean hasDataAnalysis = false;
            while (fieldNames.hasNext()) {
                if ("DataAnalysis".equals(fieldNames.next())) {
                    hasDataAnalysis = true;
                    break;
                }
            }
            if (!hasDataAnalysis) {
                throw new RuntimeException();
            }

            ArrayNode arrayNode = (ArrayNode) node.get("DataAnalysis");
            for (int i = 0; i < arrayNode.size(); i++) {
                JsonNode analysisNode = arrayNode.get(i);
                String name = analysisNode.get(COL_NAME).asText();
                int idx = analysisNode.get(COL_IDX).asInt();
                ColumnType type = ColumnType.valueOf(analysisNode.get(COL_TYPE).asText());

                JsonNode daNode = analysisNode.get(ANALYSIS);
                ColumnAnalysis dataAnalysis = om.treeToValue(daNode, ColumnAnalysis.class);

                if (type == ColumnType.Categorical) {
                    ArrayNode an = (ArrayNode) analysisNode.get(CATEGORICAL_STATE_NAMES);
                    List<String> stateNames = new ArrayList<>(an.size());
                    Iterator<JsonNode> iter = an.elements();
                    while (iter.hasNext()) {
                        stateNames.add(iter.next().asText());
                    }
                    meta.add(new CategoricalMetaData(name, stateNames));
                } else {
                    meta.add(type.newColumnMetaData(name));
                }

                analysis.add(dataAnalysis);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        Schema schema = new Schema(meta);
        return new DataAnalysis(schema, analysis);
    }

    @Deprecated //Legacy format, no longer used
    private Map<String, List<Map<String, Object>>> getJsonRepresentation() {
        Map<String, List<Map<String, Object>>> jsonRepresentation = new LinkedHashMap<>();
        List<Map<String, Object>> list = new ArrayList<>();
        jsonRepresentation.put("DataAnalysis", list);

        for (String colName : schema.getColumnNames()) {
            Map<String, Object> current = new LinkedHashMap<>();
            int idx = schema.getIndexOfColumn(colName);
            current.put(COL_NAME, colName);
            current.put(COL_IDX, idx);
            ColumnType columnType = schema.getMetaData(colName).getColumnType();
            current.put(COL_TYPE, columnType);
            if (columnType == ColumnType.Categorical) {
                current.put(CATEGORICAL_STATE_NAMES,
                                ((CategoricalMetaData) schema.getMetaData(colName)).getStateNames());
            }
            current.put(ANALYSIS, Collections.singletonMap(columnAnalysis.get(idx).getClass().getSimpleName(),
                            columnAnalysis.get(idx)));

            list.add(current);
        }

        return jsonRepresentation;
    }

    private String toJson(Map<String, List<Map<String, Object>>> jsonRepresentation) {
        ObjectMapper om = new JsonSerializer().getObjectMapper();
        try {
            return om.writeValueAsString(jsonRepresentation);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private String toYaml(Map<String, List<Map<String, Object>>> jsonRepresentation) {
        ObjectMapper om = new YamlSerializer().getObjectMapper();
        try {
            return om.writeValueAsString(jsonRepresentation);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
