package org.deeplearning4j.spark.text;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;

/**
 * @author Jeffrey Tang
 */
public class ReduceVocabFunction implements Function2<Pair<VocabCache,Long>, Pair<VocabCache,Long>, Pair<VocabCache,Long>> {
    // Function to computer the vocab to word count of each word in vocab
    public Pair<VocabCache,Long> call(Pair<VocabCache,Long> a, Pair<VocabCache,Long> b) {
        // Add InMemoryLookupCache
        InMemoryLookupCache bVocabCache = (InMemoryLookupCache)b.getFirst();
        InMemoryLookupCache aVocabCache = (InMemoryLookupCache)a.getFirst();
        Counter<String> bWordFreq = bVocabCache.getWordFrequencies();
        bWordFreq.incrementAll(aVocabCache.getWordFrequencies());
        bVocabCache.setWordFrequencies(bWordFreq);
        // Add words encountered
        Long sumWordEncountered = b.getSecond() + a.getSecond();
        return new Pair<>((VocabCache)bVocabCache, sumWordEncountered);
    }
}

