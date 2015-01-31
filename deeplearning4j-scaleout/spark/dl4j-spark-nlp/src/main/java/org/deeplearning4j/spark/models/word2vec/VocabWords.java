package org.deeplearning4j.spark.models.word2vec;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

import java.util.Collection;
import java.util.List;

/**
 * Created by agibsonccc on 1/30/15.
 */
public class VocabWords implements Function<Collection<String>,List<VocabWord>> {
    private Broadcast<VocabCache> vocab;

    public VocabWords(Broadcast<VocabCache> vocab) {
        this.vocab = vocab;
    }


    @Override
    public List<VocabWord> call(Collection<String> v1) throws Exception {
        return null;
    }
}
