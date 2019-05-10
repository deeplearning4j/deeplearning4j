/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.spark.text.functions;

import org.apache.spark.Accumulator;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.nd4j.linalg.primitives.Counter;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author Jeffrey Tang
 */
public class UpdateWordFreqAccumulatorFunction implements Function<List<String>, Pair<List<String>, AtomicLong>> {

    private Broadcast<List<String>> stopWords;
    private Accumulator<Counter<String>> wordFreqAcc;

    public UpdateWordFreqAccumulatorFunction(Broadcast<List<String>> stopWords,
                    Accumulator<Counter<String>> wordFreqAcc) {
        this.wordFreqAcc = wordFreqAcc;
        this.stopWords = stopWords;
    }

    // Function to add to word freq counter and total count of words
    @Override
    public Pair<List<String>, AtomicLong> call(List<String> lstOfWords) throws Exception {
        List<String> stops = stopWords.getValue();
        Counter<String> counter = new Counter<>();

        for (String w : lstOfWords) {
            if (w.isEmpty())
                continue;

            if (!stops.isEmpty()) {
                if (stops.contains(w)) {
                    counter.incrementCount("STOP", 1.0f);
                } else {
                    counter.incrementCount(w, 1.0f);
                }
            } else {
                counter.incrementCount(w, 1.0f);
            }
        }
        wordFreqAcc.add(counter);
        AtomicLong lstOfWordsSize = new AtomicLong(lstOfWords.size());
        return new Pair<>(lstOfWords, lstOfWordsSize);
    }
}

