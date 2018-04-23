package org.deeplearning4j.nn.conf.serde.legacyformat;

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

public class LegacyPreprocessorDeserializer extends BaseLegacyDeserializer<InputPreProcessor> {

    private static final Map<String,String> LEGACY_NAMES = new HashMap<>();

    static {

        LEGACY_NAMES.put("cnnToRnn", CnnToRnnPreProcessor.class.getName());
        LEGACY_NAMES.put("composableInput", ComposableInputPreProcessor.class.getName());
        LEGACY_NAMES.put("feedForwardToCnn", FeedForwardToCnnPreProcessor.class.getName());
        LEGACY_NAMES.put("feedForwardToRnn", FeedForwardToRnnPreProcessor.class.getName());
        LEGACY_NAMES.put("rnnToFeedForward", RnnToFeedForwardPreProcessor.class.getName());
        LEGACY_NAMES.put("rnnToCnn", RnnToCnnPreProcessor.class.getName());
        LEGACY_NAMES.put("binomialSampling", BinomialSamplingPreProcessor.class.getName());
        LEGACY_NAMES.put("unitVariance", UnitVarianceProcessor.class.getName());
        LEGACY_NAMES.put("zeroMeanAndUnitVariance", ZeroMeanAndUnitVariancePreProcessor.class.getName());
        LEGACY_NAMES.put("zeroMean",ZeroMeanPrePreProcessor.class.getName());

        //TODO: Keras preprocessors
    }


    @Override
    public Map<String, String> getLegacyNamesMap() {
        return LEGACY_NAMES;
    }

    @Override
    public ObjectMapper getLegacyJsonMapper() {
        return JsonMappers.getMapperLegacyJson();
    }
}
