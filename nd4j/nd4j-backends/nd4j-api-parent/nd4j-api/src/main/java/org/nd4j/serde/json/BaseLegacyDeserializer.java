package org.nd4j.serde.json;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * A base deserialization class used to handle deserializing of a specific class given changes from subtype wrapper
 * format to class field format.<br>
 * That is: from...<br>
 * {@literal {@code @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)}}<br>
 * {@literal {@code @JsonSubTypes(value = {@JsonSubTypes.Type(value = LossBinaryXENT.class, name = "BinaryXENT"),}...}}<br>
 * <br>
 * to<br>
 * <br>
 * {@literal {@code @JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")}}
 *
 * @param <T>  Type to deserialize
 * @author Alex Black
 */
@Slf4j
public abstract class BaseLegacyDeserializer<T> extends JsonDeserializer<T> {

    public abstract Map<String,String> getLegacyNamesMap();

    public abstract ObjectMapper getLegacyJsonMapper();

    public abstract Class<?> getDeserializedType();

    @Override
    public T deserialize(JsonParser jp, DeserializationContext deserializationContext) throws IOException {
        //Manually parse old format
        JsonNode node = jp.getCodec().readTree(jp);

        Iterator<Map.Entry<String,JsonNode>> nodes = node.fields();

        List<Map.Entry<String,JsonNode>> list = new ArrayList<>();
        while(nodes.hasNext()){
            list.add(nodes.next());
        }

        if(list.size() != 1){
            //Should only occur if field is null?
            return null;
        }

        String name = list.get(0).getKey();
        JsonNode value = list.get(0).getValue();

        Map<String,String> legacyNamesMap = getLegacyNamesMap();
        String layerClass = legacyNamesMap.get(name);
        if(layerClass == null){
            throw new IllegalStateException("Cannot deserialize " + getDeserializedType() + " with name \"" + name
                    + "\": legacy class mapping with this name is unknown");
        }

        Class<? extends T> lClass;
        try {
            lClass = (Class<? extends T>) Class.forName(layerClass);
        } catch (Exception e){
            throw new RuntimeException("Could not find class for deserialization of \"" + name + "\" of type " +
                    getDeserializedType() + ": class " + layerClass + " is not on the classpath?", e);
        }

        ObjectMapper m = getLegacyJsonMapper();

        if(m == null){
            //Should never happen, unless the user is doing something unusual
            throw new IllegalStateException("Cannot deserialize unknown subclass of type " +
                    getDeserializedType() + ": no legacy JSON mapper has been set");
        }

        String nodeAsString = value.toString();
        try {
            T t = m.readValue(nodeAsString, lClass);
            return t;
        } catch (Throwable e){
            throw new IllegalStateException("Cannot deserialize legacy JSON format of object with name \"" + name
                    + "\" of type " + getDeserializedType().getName(), e);
        }
    }



}
