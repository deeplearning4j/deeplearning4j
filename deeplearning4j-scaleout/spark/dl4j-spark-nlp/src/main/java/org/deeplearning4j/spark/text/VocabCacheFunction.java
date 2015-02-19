/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.spark.text;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.movingwindow.Util;

import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Vocab vocab function: handles word counts
 * @author Adam Gibson
 */
public class VocabCacheFunction implements Function<Pair<List<String>,Long>,Pair<VocabCache,Long>> {

    private int minWordFrequency = 5;
    private VocabCache vocab;
    private Broadcast<List<String>> stopWords;

    public VocabCacheFunction(int minWordFrequency,VocabCache vocab,Broadcast<List<String>> stopWords) {
        this.minWordFrequency = minWordFrequency;
        this.vocab = vocab;
        this.stopWords = stopWords;
    }


    @Override
    public Pair<VocabCache, Long> call(Pair<List<String>, Long> v1) throws Exception {
        Set<String> encountered = new HashSet<>();
        long wordsEncountered = v1.getSecond() + v1.getFirst().size();
        for(String token : v1.getFirst()) {
            if(stopWords.getValue().contains(token))
                token = "STOP";
            if(token.isEmpty())
                continue;

            String oldToken = token;

            if(token.isEmpty())
                token = oldToken;

            vocab.incrementWordCount(token);


            if(!encountered.contains(token)) {
                vocab.incrementDocCount(token, 1);
                encountered.add(token);
            }


            VocabWord token2;
            if(vocab.hasToken(token))
                token2 = vocab.tokenFor(token);
            else {
                token2 = new VocabWord(1.0, token);
                vocab.addToken(token2);


            }


            //note that for purposes of word frequency, the
            //internal vocab and the final vocab
            //at the class level contain the same references
            if(!Util.matchesAnyStopWord(stopWords.getValue(), token) && token != null && !token.isEmpty()) {
                if(!vocab.containsWord(token) && vocab.wordFrequency(token) >= minWordFrequency) {
                    int idx = vocab.numWords();
                    token2.setIndex(idx);
                    vocab.putVocabWord(token);
                }

                else  if(Util.matchesAnyStopWord(stopWords.getValue(),token) && token != null && !token.isEmpty()) {
                    token = "STOP";
                    if(!vocab.containsWord(token) && vocab.wordFrequency(token) >= minWordFrequency) {
                        int idx = vocab.numWords();
                        token2.setIndex(idx);
                        vocab.putVocabWord(token);
                    }


                }



            }
        }

        return new Pair<>(vocab,wordsEncountered);


    }
}
