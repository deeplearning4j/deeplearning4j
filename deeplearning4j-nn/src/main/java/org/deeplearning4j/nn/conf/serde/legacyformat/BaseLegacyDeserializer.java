package org.deeplearning4j.nn.conf.serde.legacyformat;

import org.deeplearning4j.nn.conf.graph.*;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.graph.rnn.ReverseTimeSeriesVertex;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.serde.JsonMappers;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.*;

public abstract class BaseLegacyDeserializer<T> extends JsonDeserializer<T> {

    public abstract Map<String,String> getLegacyNamesMap();

    @Override
    public T deserialize(JsonParser jp, DeserializationContext deserializationContext) throws IOException {
        //Manually parse old format
        JsonNode node = jp.getCodec().readTree(jp);

        Iterator<Map.Entry<String,JsonNode>> nodes = node.fields();
        //For legacy format, ex

        List<Map.Entry<String,JsonNode>> list = new ArrayList<>();
        while(nodes.hasNext()){
            list.add(nodes.next());
        }

        if(list.size() != 1){
            throw new IllegalStateException("Expected size 1: " + list.size());
        }

        String name = list.get(0).getKey();
        JsonNode value = list.get(0).getValue();

        String layerClass = getLegacyNamesMap().get(name);
        if(layerClass == null){
            throw new IllegalStateException("Cannot deserialize: " + name);
        }

        Class<? extends T> lClass;
        try {
            lClass = (Class<? extends T>) Class.forName(layerClass);
        } catch (Exception e){
            throw new RuntimeException(e);
        }

        ObjectMapper m = JsonMappers.getMapperLegacyJson();

        String nodeAsString = value.toString();
        T t = m.readValue(nodeAsString, lClass);
        return t;
    }



}
