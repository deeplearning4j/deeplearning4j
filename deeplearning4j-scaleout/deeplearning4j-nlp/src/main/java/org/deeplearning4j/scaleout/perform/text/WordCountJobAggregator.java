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

package org.deeplearning4j.scaleout.perform.text;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.scaleout.aggregator.JobAggregator;
import org.deeplearning4j.scaleout.job.Job;

import java.util.HashSet;
import java.util.Set;

/**
 * Word count aggregator for a vocab cache
 *
 * @author Adam Gibson
 */
public class WordCountJobAggregator implements JobAggregator {
    private VocabCache<VocabWord> vocabCache;
    public final static String MIN_WORD_FREQUENCY = "org.deeplearning4j.scaleout.perform.text.minwordfrequency";
    private int minWordFrequency = 5;

    public WordCountJobAggregator() {
        this(new InMemoryLookupCache());
    }

    public WordCountJobAggregator(VocabCache vocabCache) {
        this.vocabCache = vocabCache;
    }

    @Override
    public void accumulate(Job job) {
        Counter<String> wordCounts = (Counter<String>) job.getResult();
        Set<String> seen = new HashSet<>();
        for(String word : wordCounts.keySet()) {
            vocabCache.incrementWordCount(word,(int) wordCounts.getCount(word));
            if(!seen.contains(word)) {
                vocabCache.incrementTotalDocCount();
                vocabCache.incrementDocCount(word,1);
            }

            VocabWord token = vocabCache.tokenFor(word);
            if(token == null) {
                token = new VocabWord(wordCounts.getCount(word),word);
                vocabCache.addToken(token);

            }
            else if(vocabCache.wordFrequency(word) >= minWordFrequency) {
                //add to the vocab if it was already a token and occurred >= min word frequency times
                VocabWord vocabWord = vocabCache.wordFor(word);
                if(vocabWord == null) {
                    vocabCache.putVocabWord(word);
                }

            }
        }

    }

    @Override
    public Job aggregate() {
        Job ret =  new Job("","");
        ret.setResult(vocabCache);
        return ret;
    }

    @Override
    public void init(Configuration conf) {
        minWordFrequency = conf.getInt(MIN_WORD_FREQUENCY,5);
    }
}
