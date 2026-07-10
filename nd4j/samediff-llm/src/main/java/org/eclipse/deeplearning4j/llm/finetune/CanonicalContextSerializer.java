package org.eclipse.deeplearning4j.llm.finetune;

import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;

import java.util.Map;

/** Deterministically serializes arbitrary structured conditioning data. */
public class CanonicalContextSerializer {
    private final ObjectMapper mapper;

    public CanonicalContextSerializer() {
        mapper = new ObjectMapper();
        mapper.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        mapper.configure(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS, true);
    }

    public String serialize(Map<String, Object> context) {
        try {
            return mapper.writerWithDefaultPrettyPrinter().writeValueAsString(context);
        } catch (JsonProcessingException e) {
            throw new IllegalArgumentException("Unable to serialize teacher context", e);
        }
    }
}
