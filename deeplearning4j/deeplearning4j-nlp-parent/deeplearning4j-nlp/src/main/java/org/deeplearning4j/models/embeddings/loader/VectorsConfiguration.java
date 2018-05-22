package org.deeplearning4j.models.embeddings.loader;

import lombok.Data;
import org.apache.commons.codec.binary.Base64;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;

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
    private int batchSize = 512;
    private int iterations = 1;
    private int epochs = 1;
    private int window = 5;
    private long seed;
    private double negative = 0.0d;
    private boolean useHierarchicSoftmax = true;
    private double sampling = 0.0d;
    private int learningRateDecayWords;
    private int[] variableWindows;

    private boolean hugeModelExpected = false;
    private boolean useUnknown = false;

    private int scavengerActivationThreshold = 2000000;
    private int scavengerRetentionDelay = 3;

    private String elementsLearningAlgorithm;
    private String sequenceLearningAlgorithm;
    private String modelUtils;

    private String tokenizerFactory;
    private String tokenPreProcessor;

    // this is one-off configuration value specially for NGramTokenizerFactory
    private int nGram;

    private String UNK = "UNK";
    private String STOP = "STOP";

    private Collection<String> stopList = new ArrayList<>();

    // overall model info
    private int vocabSize;

    // paravec-specific option
    private boolean trainElementsVectors = true;
    private boolean trainSequenceVectors = true;
    private boolean allowParallelTokenization = false;
    private boolean preciseWeightInit = false;

    private static ObjectMapper mapper;
    private static final Object lock = new Object();

    private static ObjectMapper mapper() {
        if (mapper == null) {
            synchronized (lock) {
                if (mapper == null) {
                    mapper = new ObjectMapper();
                    mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
                    mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
                    mapper.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
                    return mapper;
                }
            }
        }
        return mapper;
    }

    public String toJson() {
        ObjectMapper mapper = mapper();
        try {
            /*
                we need JSON as single line to save it at first line of the CSV model file
                That's ugly method, but its way more memory-friendly then loading whole 10GB json file just to create another 10GB memory array.
            */
            return mapper.writeValueAsString(this);
        } catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    public String toEncodedJson() {
        Base64 base64 = new Base64(Integer.MAX_VALUE);
        try {
            return base64.encodeAsString(this.toJson().getBytes("UTF-8"));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static VectorsConfiguration fromJson(String json) {
        ObjectMapper mapper = mapper();
        try {
            VectorsConfiguration ret = mapper.readValue(json, VectorsConfiguration.class);
            return ret;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
