package org.deeplearning4j.models.word2vec.wordstore;

import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import lombok.Data;
import lombok.NonNull;
import lombok.ToString;

import java.io.IOException;
import java.io.Serializable;

/**
 * Simplified version of VocabWord.
 * Used only for w2v vocab building routines
 *
 * @author raver119@gmail.com
 */

@Data
public class VocabularyWord implements Serializable {
    @NonNull
    private String word;
    private int count = 1;

    // these fileds are used for vocab serialization/deserialization only. Usual runtime value is null.
    private double[] syn0;
    private double[] syn1;
    private double[] syn1Neg;
    private double[] historicalGradient;

    // There's no reasons to save HuffmanNode data, it will be recalculated after deserialization
    private transient HuffmanNode huffmanNode;

    // empty constructor is required for proper deserialization
    public VocabularyWord() {

    }

    public VocabularyWord(@NonNull String word) {
        this.word = word;
    }

    /*
        since scavenging mechanics are targeting low-freq words, byte values is definitely enough.
        please note, array is initialized outside of this scope, since it's only holder, no logic involved inside this class
      */
    private transient byte[] frequencyShift;
    private transient byte retentionStep;

    // special mark means that this word should NOT be affected by minWordFrequency setting
    private boolean special = false;

    public void incrementCount() {
        this.count++;
    }

    public void incrementRetentionStep() {
        this.retentionStep += 1;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        VocabularyWord word1 = (VocabularyWord) o;

        return word != null ? word.equals(word1.word) : word1.word == null;

    }

    @Override
    public int hashCode() {
        return word != null ? word.hashCode() : 0;
    }

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
            */
            return mapper.writeValueAsString(this);
        } catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    public static VocabularyWord fromJson(String json) {
        ObjectMapper mapper = mapper();
        try {
            VocabularyWord ret =  mapper.readValue(json, VocabularyWord.class);
            return ret;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
