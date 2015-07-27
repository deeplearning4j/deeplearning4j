package org.deeplearning4j.spark.text;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;

import java.util.Map;

/**
 * @author Jeffrey Tang
 */
public class ReduceVocabFunction implements Function2<Pair<VocabCache,Long>, Pair<VocabCache,Long>, Pair<VocabCache,Long>> {
    // Function to computer the vocab to word count of each word in vocab
    public Pair<VocabCache,Long> call(Pair<VocabCache,Long> a, Pair<VocabCache,Long> b) {
        // Add InMemoryLookupCache
        InMemoryLookupCache bVocabCache = (InMemoryLookupCache)b.getFirst();
        InMemoryLookupCache aVocabCache = (InMemoryLookupCache)a.getFirst();
        // Add word frequency
        Counter<String> bWordFreq = bVocabCache.getWordFrequencies();
        bWordFreq.setDeflt(0.0);
        bWordFreq.incrementAll(aVocabCache.getWordFrequencies());
        bVocabCache.setWordFrequencies(bWordFreq);
        // Add token
        Map<String, VocabWord> bToken = bVocabCache.getTokens();
        Map<String, VocabWord> aToken = aVocabCache.getTokens();
        bToken.putAll(aToken);
        bVocabCache.setVocabs(bToken);
        // Add words encountered
        Long sumWordEncountered = b.getSecond() + a.getSecond();
        return new Pair<>((VocabCache)bVocabCache, sumWordEncountered);
    }
}

