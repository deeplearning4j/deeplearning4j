package org.nd4j.interceptor.parser;

import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.type.TypeReference;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;

import java.io.IOException;
import java.util.Map;

public class SourceCodeIndexComparatorDeserializer extends JsonDeserializer<SourceCodeIndexComparator> {
    @Override
    public SourceCodeIndexComparator deserialize(JsonParser p, DeserializationContext ctxt) throws IOException {
        SourceCodeIndexer index1 = p.readValueAs(SourceCodeIndexer.class);
        SourceCodeIndexer index2 = p.readValueAs(SourceCodeIndexer.class);
        Map<SourceCodeLine, SourceCodeLine> comparisonResult = p.readValueAs(new TypeReference<Map<SourceCodeLine, SourceCodeLine>>() {});
        Map<SourceCodeLine, SourceCodeLine> reverseComparisonResult = p.readValueAs(new TypeReference<Map<SourceCodeLine, SourceCodeLine>>() {});

        return SourceCodeIndexComparator.builder()
                .index1(index1)
                .index2(index2)
                .comparisonResult(comparisonResult)
                .reverseComparisonResult(reverseComparisonResult)
                .build();
    }
}