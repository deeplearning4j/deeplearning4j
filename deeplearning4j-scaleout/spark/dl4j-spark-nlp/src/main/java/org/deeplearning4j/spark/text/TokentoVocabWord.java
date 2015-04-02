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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Handles converting tokens to vocab words based
 * on a given vocab
 *
 * @author Adam Gibson
 */
public class TokentoVocabWord implements Function<Pair<List<String>,Long>,Pair<List<VocabWord>,AtomicLong>> {
    private  Broadcast<VocabCache> vocab;
    private AtomicLong lastSeen = new AtomicLong(0);

    public TokentoVocabWord(Broadcast<VocabCache> vocab) {
        this.vocab = vocab;
    }

    @Override
    public Pair<List<VocabWord>,AtomicLong> call(Pair<List<String>,Long> v1) throws Exception {
        List<VocabWord> ret = new ArrayList<>();
        for(String s : v1.getFirst())
            ret.add(vocab.getValue().wordFor(s));
        return new Pair<>(ret,new AtomicLong(v1.getSecond()));
    }
}
