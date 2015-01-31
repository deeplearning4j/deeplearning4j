package org.deeplearning4j.spark.text;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Handles converting tokens to vocab words based
 * on a given vo
 */
public class TokentoVocabWord implements Function<Pair<List<String>,Long>,Pair<List<VocabWord>,AtomicLong>> {
    private Broadcast<VocabCache> vocab;

    public TokentoVocabWord(Broadcast<VocabCache> vocab) {
        this.vocab = vocab;
    }

    @Override
    public Pair<List<VocabWord>,AtomicLong> call(Pair<List<String>,Long> v1) throws Exception {
        List<VocabWord> ret = new ArrayList<>();
        for(String s : v1.getFirst())
        ret.add(vocab.getValue().wordFor(s));
        return new Pair<>(ret,new AtomicLong(0));
    }
}
