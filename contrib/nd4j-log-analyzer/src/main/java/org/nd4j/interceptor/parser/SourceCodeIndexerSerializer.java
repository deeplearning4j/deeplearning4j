package org.nd4j.interceptor.parser;

import org.nd4j.shade.guava.collect.Table;
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class SourceCodeIndexerSerializer extends JsonSerializer<SourceCodeIndexer> {


    @Override
    public void serialize(SourceCodeIndexer sourceCodeIndexer, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException {
        jsonGenerator.writeStartObject();

        // Convert the Table to a Map
        Map<String, Map<Integer, SourceCodeLine>> map = new HashMap<>();
        for (Table.Cell<String, Integer, SourceCodeLine> cell : sourceCodeIndexer.getIndex().cellSet()) {
            map.putIfAbsent(cell.getRowKey(), new HashMap<>());
            map.get(cell.getRowKey()).put(cell.getColumnKey(), cell.getValue());
        }

        jsonGenerator.writeObjectField("index", map);
        jsonGenerator.writeEndObject();
    }
}