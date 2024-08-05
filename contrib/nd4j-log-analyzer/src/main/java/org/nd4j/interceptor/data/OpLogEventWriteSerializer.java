package org.nd4j.interceptor.data;

import org.json.JSONArray;
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.shade.jackson.databind.SerializerProvider;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class OpLogEventWriteSerializer extends JsonSerializer<OpLogEvent> {
    @Override
    public void serialize(OpLogEvent value, JsonGenerator gen, SerializerProvider serializers) throws IOException {
        gen.useDefaultPrettyPrinter();

        gen.writeStartObject();

        gen.writeFieldName("opName");
        gen.writeString(value.getOpName());

        gen.writeFieldName("inputs");
        serializeInputOutput(value.getInputs(), gen);

        gen.writeFieldName("outputs");
        serializeInputOutput(value.getOutputs(), gen);

        gen.writeFieldName("stackTrace");
        serializeStackTrace(value.getStackTrace().split("\n"), gen);

        gen.writeFieldName("eventId");
        gen.writeNumber(value.getEventId());
        gen.writeEndObject();
    }

    private void serializeInputOutput(Map<Integer, String> valuesMap, JsonGenerator gen) throws IOException {
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.enable(SerializationFeature.INDENT_OUTPUT);
        Map<String, Object> write = new LinkedHashMap<>();
        for (Map.Entry<Integer, String> entry : valuesMap.entrySet()) {
            String item = entry.getValue();
            try {
                JSONArray jsonArray = new JSONArray(item);
                write.put(String.valueOf(entry.getKey()), jsonArray.toString(2));
            } catch (Exception e) {
                // scalar cases
                write.put(String.valueOf(entry.getKey()), item);
            }
        }
        gen.writeStartObject();
        for (Map.Entry<String, Object> entry : write.entrySet()) {
            gen.writeFieldName(entry.getKey());
            if (entry.getValue() instanceof Map) {
                gen.writeStartObject();
                @SuppressWarnings("unchecked")
                Map<String, Object> map = (Map<String, Object>) entry.getValue();
                for (Map.Entry<String, Object> mapEntry : map.entrySet()) {
                    gen.writeFieldName(mapEntry.getKey());
                    if (mapEntry.getValue() instanceof Map) {
                        gen.writeStartObject();
                        @SuppressWarnings("unchecked")
                        Map<String, Object> innerMap = (Map<String, Object>) mapEntry.getValue();
                        for (Map.Entry<String, Object> innerEntry : innerMap.entrySet()) {
                            gen.writeFieldName(innerEntry.getKey());
                            gen.writeNumber(((Double) innerEntry.getValue()).doubleValue());
                        }
                        gen.writeEndObject();
                    } else {
                        gen.writeString((String) mapEntry.getValue());
                    }
                }
                gen.writeEndObject();
            } else {
                gen.writeString((String) entry.getValue());
            }
        }
        gen.writeEndObject();
    }

    private void serializeStackTrace(String[] stackTrace, JsonGenerator gen) throws IOException {
        if(stackTrace.length == 1) {
            JSONArray jsonArray = new JSONArray(stackTrace[0]);
            gen.writeRawValue(jsonArray.toString(2));
        } else {
            JSONArray jsonArray = new JSONArray();
            for (String item : stackTrace) {
                jsonArray.put(item);
            }
            gen.writeRawValue(jsonArray.toString(2));
        }

    }
}