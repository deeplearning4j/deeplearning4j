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

package org.deeplearning4j.spark.text;

import org.apache.spark.Accumulator;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import java.util.List;

/**
 * @author Jeffrey Tang
 */
public class VocabCacheFunction implements VoidFunction<Pair<List<String>, Long>> {

    private Broadcast<List<String>> stopWords;
    private Accumulator<Counter<String>> wordFreqAcc;
    private Accumulator<Double> wordCountAcc;

    //Getters
    public Accumulator<Counter<String>> getWordFreqAcc() {
        return wordFreqAcc;
    }

    public Accumulator<Double> getWordCountAcc() {
        return wordCountAcc;
    }
    //

    public VocabCacheFunction(Broadcast<List<String>> stopWords,

                              Accumulator<Counter<String>> wordFreqAcc,
                              Accumulator<Double> wordCountAcc) {
        this.wordFreqAcc = wordFreqAcc;
        this.wordCountAcc = wordCountAcc;
        this.stopWords = stopWords;
    }

    // Function to add to word freq counter and total count of words
    @Override
    public void call(Pair<List<String>, Long> pair) throws Exception {
        // Add the count of the sentence to the global count of all words
        Long sentenceWordCount = pair.getSecond();
        wordCountAcc.add((double)sentenceWordCount);
        // Set up the list of words and the stop words and the counter
        List<String> lstOfWords = pair.getFirst();
        List<String> stops = stopWords.getValue();
        Counter<String> counter = new Counter<>();

        for (String w : lstOfWords) {
            if(w.isEmpty())
                continue;

            if (stops.contains(w)) {
                counter.incrementCount("STOP", 1.0);
            } else {
                counter.incrementCount(w, 1.0);
            }
        }
        wordFreqAcc.add(counter);
    }
}

