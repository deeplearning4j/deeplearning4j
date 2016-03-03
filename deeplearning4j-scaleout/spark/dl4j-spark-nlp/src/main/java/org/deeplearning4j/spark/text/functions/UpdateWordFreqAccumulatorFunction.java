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

package org.deeplearning4j.spark.text.functions;

import org.apache.spark.Accumulator;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author Jeffrey Tang
 */
public class UpdateWordFreqAccumulatorFunction implements Function<List<String>, Pair<List<String>, AtomicLong>> {

    private Broadcast<List<String>> stopWords;
    private Accumulator<Counter<String>> wordFreqAcc;

    public UpdateWordFreqAccumulatorFunction(Broadcast<List<String>> stopWords, Accumulator<Counter<String>> wordFreqAcc) {
        this.wordFreqAcc = wordFreqAcc;
        this.stopWords = stopWords;
    }

    // Function to add to word freq counter and total count of words
    @Override
    public Pair<List<String>, AtomicLong> call(List<String> lstOfWords) throws Exception {
        List<String> stops = stopWords.getValue();
        Counter<String> counter = new Counter<>();

        for (String w : lstOfWords) {
            if(w.isEmpty())
                continue;

            if (!stops.isEmpty()) {
                if (stops.contains(w)) {
                    counter.incrementCount("STOP", 1.0);
                } else {
                    counter.incrementCount(w, 1.0);
                }
            }  else {
                counter.incrementCount(w, 1.0);
            }
        }
        wordFreqAcc.add(counter);
        AtomicLong lstOfWordsSize = new AtomicLong(lstOfWords.size());
        return new Pair<>(lstOfWords, lstOfWordsSize);
    }
}

