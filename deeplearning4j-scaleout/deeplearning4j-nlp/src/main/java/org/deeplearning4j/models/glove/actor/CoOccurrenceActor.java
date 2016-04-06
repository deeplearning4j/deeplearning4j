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

package org.deeplearning4j.models.glove.actor;

import akka.actor.UntypedActor;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.CounterMap;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 *
 * Co occurrence actor
 *
 * @author Adam Gibson
 */
public class CoOccurrenceActor extends UntypedActor {
    private TokenizerFactory tokenizerFactory;
    private int windowSize = 5;
    private VocabCache<? extends SequenceElement> cache;
    private CounterMap<String,String> coOCurreneCounts = new CounterMap<>();
    private Counter<Integer> occurrenceAllocations;
    private AtomicInteger processed;
    private boolean symmetric = true;
    private static final Logger log = LoggerFactory.getLogger(CoOccurrenceActor.class);

    public CoOccurrenceActor(AtomicInteger processed,TokenizerFactory tokenizerFactory, int windowSize, VocabCache cache, CounterMap<String,String> coOCurreneCounts,boolean symmetric,Counter<Integer> occurrenceAllocations) {
        this.processed = processed;
        this.tokenizerFactory = tokenizerFactory;
        this.windowSize = windowSize;
        this.cache = cache;
        this.coOCurreneCounts = coOCurreneCounts;
        this.symmetric = symmetric;
        this.occurrenceAllocations = occurrenceAllocations;
    }

    @Override
    public void onReceive(Object message) throws Exception {
        if(message instanceof SentenceWork) {
            SentenceWork work = (SentenceWork) message;
            String s =  work.getSentence();
            List<String> tokens = tokenizerFactory.create(s).getTokens();
            for(int i = 0; i < tokens.size(); i++) {
                int wordIdx = cache.indexOf(tokens.get(i));
                if (wordIdx < 0) continue;
                String w1 = cache.wordFor(tokens.get(i)).getLabel();

                if(w1.equals(Glove.DEFAULT_UNK))
                    continue;
                int windowStop = Math.min(i + windowSize + 1,tokens.size());
                for(int j = i; j < windowStop; j++) {
                    int otherWord = cache.indexOf(tokens.get(j));
                    if (otherWord < 0) continue;
                    String w2 = cache.wordFor(tokens.get(j)).getLabel();
                    if(w2.equals(Glove.DEFAULT_UNK) || otherWord == wordIdx)
                        continue;
                    if(wordIdx < otherWord) {
                        coOCurreneCounts.incrementCount(tokens.get(i), tokens.get(j), 1.0 / (j - i + Nd4j.EPS_THRESHOLD));
                        occurrenceAllocations.incrementCount(work.getId(),1.0);
                        if(symmetric) {
                            coOCurreneCounts.incrementCount(tokens.get(j), tokens.get(i), 1.0 / (j - i + Nd4j.EPS_THRESHOLD));
                            occurrenceAllocations.incrementCount(work.getId(),1.0);

                        }

                    }
                    else {
                        coOCurreneCounts.incrementCount(tokens.get(j),tokens.get(i), 1.0 / (j - i + Nd4j.EPS_THRESHOLD));
                        occurrenceAllocations.incrementCount(work.getId(),1.0);

                        if(symmetric) {
                            coOCurreneCounts.incrementCount(tokens.get(i), tokens.get(j), 1.0 / (j - i + Nd4j.EPS_THRESHOLD));
                            occurrenceAllocations.incrementCount(work.getId(),1.0);

                        }
                    }


                }
            }

            processed.incrementAndGet();

        }

        else
            unhandled(message);
    }
}

