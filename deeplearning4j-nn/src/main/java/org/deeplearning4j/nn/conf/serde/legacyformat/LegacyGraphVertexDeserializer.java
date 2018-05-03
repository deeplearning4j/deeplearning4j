package org.deeplearning4j.nn.conf.serde.legacyformat;

import lombok.NonNull;
import org.deeplearning4j.nn.conf.graph.*;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.graph.rnn.ReverseTimeSeriesVertex;
import org.deeplearning4j.nn.conf.serde.JsonMappers;
import org.nd4j.serde.json.BaseLegacyDeserializer;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Deserializer for GraphVertex JSON in legacy format - see {@link BaseLegacyDeserializer}
 *
 * @author Alex Black
 */
public class LegacyGraphVertexDeserializer extends BaseLegacyDeserializer<GraphVertex> {

    private static final Map<String,String> LEGACY_NAMES = new HashMap<>();

    static {


        List<Class<? extends GraphVertex>> cList = Arrays.asList(
                //All of these vertices had the legacy format name the same as the simple class name
                MergeVertex.class,
                SubsetVertex.class,
                LayerVertex.class,
                LastTimeStepVertex.class,
                ReverseTimeSeriesVertex.class,
                DuplicateToTimeSeriesVertex.class,
                PreprocessorVertex.class,
                StackVertex.class,
                UnstackVertex.class,
                L2Vertex.class,
                ScaleVertex.class,
                L2NormalizeVertex.class,
                //These did not previously have a subtype annotation - they use default (which is simple class name)
                ElementWiseVertex.class,
                PoolHelperVertex.class,
                ReshapeVertex.class,
                ShiftVertex.class);

        for(Class<?> c : cList){
            LEGACY_NAMES.put(c.getSimpleName(), c.getName());
        }
    }


    @Override
    public Map<String, String> getLegacyNamesMap() {
        return LEGACY_NAMES;
    }

    @Override
    public ObjectMapper getLegacyJsonMapper() {
//        return JsonMappers.getMapperLegacyJson();
        return JsonMappers.getJsonMapperLegacyFormatVertex();
    }

    @Override
    public Class<?> getDeserializedType() {
        return GraphVertex.class;
    }

    public static void registerLegacyClassDefaultName(@NonNull Class<? extends GraphVertex> clazz){
        registerLegacyClassSpecifiedName(clazz.getSimpleName(), clazz);
    }

    public static void registerLegacyClassSpecifiedName(@NonNull String name, @NonNull Class<? extends GraphVertex> clazz){
        LEGACY_NAMES.put(name, clazz.getName());
    }
}
