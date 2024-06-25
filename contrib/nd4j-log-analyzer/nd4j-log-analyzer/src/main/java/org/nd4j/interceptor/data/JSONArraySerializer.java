package org.nd4j.interceptor.data;

import org.json.JSONArray;
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;
import org.nd4j.shade.jackson.databind.module.SimpleModule;

import java.io.IOException;

public class JSONArraySerializer extends JsonSerializer<JSONArray> {

    @Override
    public void serialize(JSONArray value, JsonGenerator gen, SerializerProvider serializers) throws IOException {
        gen.writeStartArray();
        for (int i = 0; i < value.length(); i++) {
            Object item = value.opt(i);
            if (item == null) {
                gen.writeNull();
            } else if (item instanceof Boolean) {
                gen.writeBoolean((Boolean) item);
            } else if (item instanceof Integer) {
                gen.writeNumber((Integer) item);
            } else if (item instanceof Long) {
                gen.writeNumber((Long) item);
            } else if (item instanceof Double) {
                gen.writeNumber((Double) item);
            } else if (item instanceof String) {
                gen.writeString((String) item);
            } else if (item instanceof JSONArray) {
                serialize((JSONArray) item, gen, serializers);
            } else {
                gen.writeObject(item.toString());
            }
        }
        gen.writeEndArray();
    }

    public static class JSONArraySerializerModule extends SimpleModule {
        public JSONArraySerializerModule() {
            addSerializer(JSONArray.class, new JSONArraySerializer());
        }
    }
}