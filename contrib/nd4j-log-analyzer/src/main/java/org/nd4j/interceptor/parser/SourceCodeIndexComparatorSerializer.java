package org.nd4j.interceptor.parser;

import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;

import java.io.IOException;

public class SourceCodeIndexComparatorSerializer extends JsonSerializer<SourceCodeIndexComparator> {
    @Override
    public void serialize(SourceCodeIndexComparator value, JsonGenerator gen, SerializerProvider serializers) throws IOException {
        gen.writeStartObject();
        gen.writeObjectField("index1", value.getIndex1());
        gen.writeObjectField("index2", value.getIndex2());
        gen.writeObjectField("comparisonResult", value.getComparisonResult());
        gen.writeObjectField("reverseComparisonResult", value.getReverseComparisonResult());
        gen.writeEndObject();
    }
}