package org.nd4j.interceptor.parser;

import org.nd4j.shade.guava.collect.HashBasedTable;
import org.nd4j.shade.guava.collect.Table;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;

import java.io.IOException;
import java.util.Map;

public class SourceCodeIndexerDeserializer extends JsonDeserializer<SourceCodeIndexer> {

    @Override
    public SourceCodeIndexer deserialize(JsonParser jsonParser, DeserializationContext deserializationContext) throws IOException {
        JsonNode node = jsonParser.getCodec().readTree(jsonParser);
        SourceCodeIndexer sourceCodeIndexer = new SourceCodeIndexer();

        // Load the data as a Map
        Map<String, Map<Integer, SourceCodeLine>> map = jsonParser.getCodec().treeToValue(node.get("index"), Map.class);

        // Convert the Map to a Table
        Table<String, Integer, SourceCodeLine> table = HashBasedTable.create();
        for (Map.Entry<String, Map<Integer, SourceCodeLine>> row : map.entrySet()) {
            for (Map.Entry<Integer, SourceCodeLine> column : row.getValue().entrySet()) {
                table.put(row.getKey(), column.getKey(), column.getValue());
            }
        }

        sourceCodeIndexer.setIndex(table);
        return sourceCodeIndexer;
    }
}