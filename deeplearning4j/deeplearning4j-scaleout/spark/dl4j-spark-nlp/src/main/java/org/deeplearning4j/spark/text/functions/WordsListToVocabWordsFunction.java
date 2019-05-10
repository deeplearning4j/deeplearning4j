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

import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author jeffreytang
 */
public class WordsListToVocabWordsFunction implements Function<Pair<List<String>, AtomicLong>, List<VocabWord>> {

    Broadcast<VocabCache<VocabWord>> vocabCacheBroadcast;

    public WordsListToVocabWordsFunction(Broadcast<VocabCache<VocabWord>> vocabCacheBroadcast) {
        this.vocabCacheBroadcast = vocabCacheBroadcast;
    }

    @Override
    public List<VocabWord> call(Pair<List<String>, AtomicLong> pair) throws Exception {
        List<String> wordsList = pair.getFirst();
        List<VocabWord> vocabWordsList = new ArrayList<>();
        VocabCache<VocabWord> vocabCache = vocabCacheBroadcast.getValue();
        for (String s : wordsList) {
            if (vocabCache.containsWord(s)) {
                VocabWord word = vocabCache.wordFor(s);

                vocabWordsList.add(word);
            } else if (vocabCache.containsWord("UNK")) {
                VocabWord word = vocabCache.wordFor("UNK");

                vocabWordsList.add(word);
            }
        }
        return vocabWordsList;
    }
}

