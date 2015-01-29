package org.deeplearning4j.spark.text;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.movingwindow.Util;

import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Vocab cache function: handles word counts
 * @author Adam Gibson
 */
public class VocabCacheFunction implements Function<Triple<Collection<String>,VocabCache,Long>,Pair<VocabCache,Long>> {

    private int minWordFrequency = 5;
    private Broadcast<List<String>> stopWords;

    public VocabCacheFunction(int minWordFrequency,Broadcast<List<String>> stopWords) {
        this.minWordFrequency = minWordFrequency;
        this.stopWords = stopWords;
    }


    @Override
    public Pair<VocabCache, Long> call(Triple<Collection<String>, VocabCache, Long> v1) throws Exception {
        Set<String> encountered = new HashSet<>();
        long wordsEncountered = v1.getThird() + v1.getFirst().size();
        VocabCache cache = v1.getSecond();
        for(String token : v1.getFirst()) {
            if(stopWords.getValue().contains(token))
                token = "STOP";
            if(token.isEmpty())
                continue;

            String oldToken = token;

            if(token.isEmpty())
                token = oldToken;

            cache.incrementWordCount(token);


            if(!encountered.contains(token)) {
                cache.incrementDocCount(token,1);
                encountered.add(token);
            }


            VocabWord token2;
            if(cache.hasToken(token))
                token2 = cache.tokenFor(token);
            else {
                token2 = new VocabWord(1.0, token);
                cache.addToken(token2);


            }


            //note that for purposes of word frequency, the
            //internal vocab and the final vocab
            //at the class level contain the same references
            if(!Util.matchesAnyStopWord(stopWords.getValue(), token) && token != null && !token.isEmpty()) {
                if(!cache.containsWord(token) && cache.wordFrequency(token) >= minWordFrequency) {
                    int idx = cache.numWords();
                    token2.setIndex(idx);
                    cache.putVocabWord(token);
                }

                else  if(Util.matchesAnyStopWord(stopWords.getValue(),token) && token != null && !token.isEmpty()) {
                    token = "STOP";
                    if(!cache.containsWord(token) && cache.wordFrequency(token) >= minWordFrequency) {
                        int idx = cache.numWords();
                        token2.setIndex(idx);
                        cache.putVocabWord(token);
                    }


                }



            }
        }

        return new Pair<>(cache,wordsEncountered);


    }
}
