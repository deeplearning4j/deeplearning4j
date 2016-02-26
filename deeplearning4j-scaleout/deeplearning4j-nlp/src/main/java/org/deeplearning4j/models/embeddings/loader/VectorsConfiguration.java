package org.deeplearning4j.models.embeddings.loader;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import lombok.Data;


import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * This is simple bean/POJO for Word2Vec persistence handling.
 * It holds whole w2v model configuration info, except of TokenizerFactory and SentenceIterator, since they are not intended for serialization, and specified at run-time.
 *
 *  @author raver119@gmail.com
 */
@Data
public class VectorsConfiguration implements Serializable {

    // word2vec params
    private int minWordFrequency = 5;
    private double learningRate = 0.025;
    private double minLearningRate = 0.0001;
    private int layersSize = 200;
    private boolean useAdaGrad = false;
    private int batchSize = 1000;
    private int iterations = 1;
    private int epochs = 1;
    private int window = 5;
    private long seed;
    private double negative = 0.0d;
    private double sampling = 0.0d;
    private int learningRateDecayWords;

    private boolean hugeModelExpected = false;
    private int scavengerActivationThreshold = 2000000;
    private int scavengerRetentionDelay = 3;

    private String elementsLearningAlgorithm;
    private String sequenceLearningAlgorithm;

    private List<String> stopList = new ArrayList<>();

    // overall model info
    private int vocabSize;

    private static ObjectMapper mapper() {
        /*
              DO NOT ENABLE INDENT_OUTPUT FEATURE
              we need THIS json to be single-line
          */
        ObjectMapper ret = new ObjectMapper();
        ret.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        ret.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        ret.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        return ret;
    }

    public String toJson() {
        ObjectMapper mapper = mapper();
        try {
            /*
                we need JSON as single line to save it at first line of the CSV model file
                That's ugly method, but its way more memory-friendly then loading whole 10GB json file just to create another 10GB memory array.
            */
            return mapper.writeValueAsString(this);
        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    public static VectorsConfiguration fromJson(String json) {
        ObjectMapper mapper = mapper();
        try {
            VectorsConfiguration ret =  mapper.readValue(json, VectorsConfiguration.class);
            return ret;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
