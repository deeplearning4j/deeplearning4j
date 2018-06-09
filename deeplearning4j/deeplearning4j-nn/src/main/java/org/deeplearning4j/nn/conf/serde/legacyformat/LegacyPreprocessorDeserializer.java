package org.deeplearning4j.nn.conf.serde.legacyformat;

import lombok.NonNull;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.graph.*;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.graph.rnn.ReverseTimeSeriesVertex;
import org.deeplearning4j.nn.conf.preprocessor.*;
import org.deeplearning4j.nn.conf.serde.JsonMappers;
import org.nd4j.serde.json.BaseLegacyDeserializer;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Deserializer for InputPreProcessor JSON in legacy format - see {@link BaseLegacyDeserializer}
 *
 * @author Alex Black
 */
public class LegacyPreprocessorDeserializer extends BaseLegacyDeserializer<InputPreProcessor> {

    private static final Map<String,String> LEGACY_NAMES = new HashMap<>();

    static {
        LEGACY_NAMES.put("cnnToFeedForward", CnnToFeedForwardPreProcessor.class.getName());
        LEGACY_NAMES.put("cnnToRnn", CnnToRnnPreProcessor.class.getName());
        LEGACY_NAMES.put("composableInput", ComposableInputPreProcessor.class.getName());
        LEGACY_NAMES.put("feedForwardToCnn", FeedForwardToCnnPreProcessor.class.getName());
        LEGACY_NAMES.put("feedForwardToRnn", FeedForwardToRnnPreProcessor.class.getName());
        LEGACY_NAMES.put("rnnToFeedForward", RnnToFeedForwardPreProcessor.class.getName());
        LEGACY_NAMES.put("rnnToCnn", RnnToCnnPreProcessor.class.getName());

        //Keras preprocessors: they defaulted to class simple name
        LEGACY_NAMES.put("KerasFlattenRnnPreprocessor","org.deeplearning4j.nn.modelimport.keras.preprocessors.KerasFlattenRnnPreprocessor");
        LEGACY_NAMES.put("ReshapePreprocessor","org.deeplearning4j.nn.modelimport.keras.preprocessors.ReshapePreprocessor");
        LEGACY_NAMES.put("TensorFlowCnnToFeedForwardPreProcessor","org.deeplearning4j.nn.modelimport.keras.preprocessors.TensorFlowCnnToFeedForwardPreProcessor");
    }


    @Override
    public Map<String, String> getLegacyNamesMap() {
        return LEGACY_NAMES;
    }

    @Override
    public ObjectMapper getLegacyJsonMapper() {
//        return JsonMappers.getMapperLegacyJson();
        return JsonMappers.getJsonMapperLegacyFormatPreproc();
    }

    @Override
    public Class<?> getDeserializedType() {
        return InputPreProcessor.class;
    }

    public static void registerLegacyClassDefaultName(@NonNull Class<? extends InputPreProcessor> clazz){
        registerLegacyClassSpecifiedName(clazz.getSimpleName(), clazz);
    }

    public static void registerLegacyClassSpecifiedName(@NonNull String name, @NonNull Class<? extends InputPreProcessor> clazz){
        LEGACY_NAMES.put(name, clazz.getName());
    }
}
