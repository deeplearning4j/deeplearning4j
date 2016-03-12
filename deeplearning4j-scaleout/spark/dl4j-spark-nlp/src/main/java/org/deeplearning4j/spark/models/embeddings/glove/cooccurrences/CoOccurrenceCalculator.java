/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.spark.models.embeddings.glove.cooccurrences;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.CounterMap;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Calculate co occurrences based on tokens
 *
 * @author Adam Gibson
 */
public class CoOccurrenceCalculator implements Function<Pair<List<String>,AtomicLong>,CounterMap<String,String>> {
    private boolean symmetric = false;
    private Broadcast<VocabCache<VocabWord>> vocab;
    private int windowSize = 5;

    public CoOccurrenceCalculator(boolean symmetric, Broadcast<VocabCache<VocabWord>> vocab, int windowSize) {
        this.symmetric = symmetric;
        this.vocab = vocab;
        this.windowSize = windowSize;
    }


    @Override
    public CounterMap<String, String> call(Pair<List<String>,AtomicLong> pair) throws Exception {
        List<String> sentence = pair.getFirst();
        CounterMap<String,String> coOCurreneCounts = new CounterMap<>();
        VocabCache vocab = this.vocab.value();
        for(int i = 0; i < sentence.size(); i++) {
            int wordIdx = vocab.indexOf(sentence.get(i));
            String w1 = ((VocabWord) vocab.wordFor(sentence.get(i))).getWord();

            if(wordIdx < 0) // || w1.equals(Glove.UNK))
                continue;
            int windowStop = Math.min(i + windowSize + 1,sentence.size());
            for(int j = i; j < windowStop; j++) {
                int otherWord = vocab.indexOf(sentence.get(j));
                String w2 = ((VocabWord) vocab.wordFor(sentence.get(j))).getWord();
                if(vocab.indexOf(sentence.get(j)) < 0) // || w2.equals(Glove.UNK))
                    continue;

                if(otherWord == wordIdx)
                    continue;
                if(wordIdx < otherWord) {
                    coOCurreneCounts.incrementCount(sentence.get(i), sentence.get(j), 1.0 / (j - i + Nd4j.EPS_THRESHOLD));
                    if(symmetric)
                        coOCurreneCounts.incrementCount(sentence.get(j), sentence.get(i), 1.0 / (j - i + Nd4j.EPS_THRESHOLD));



                }
                else {
                    coOCurreneCounts.incrementCount(sentence.get(j),sentence.get(i), 1.0 / (j - i + Nd4j.EPS_THRESHOLD));
                    if(symmetric)
                        coOCurreneCounts.incrementCount(sentence.get(i), sentence.get(j), 1.0 / (j - i + Nd4j.EPS_THRESHOLD));


                }


            }
        }
        return coOCurreneCounts;
    }
}
