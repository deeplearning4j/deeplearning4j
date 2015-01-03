package org.deeplearning4j.models.glove.actor;

import akka.actor.UntypedActor;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.CounterMap;
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
    private VocabCache cache;
    private CounterMap<String,String> coOCurreneCounts = new CounterMap<>();
    private Counter<Integer> occurrenceAllocations;
    private AtomicInteger processed;
    private boolean symmetric = true;
    private static Logger log = LoggerFactory.getLogger(CoOccurrenceActor.class);

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
                String w1 = cache.wordFor(tokens.get(i)).getWord();

                if(wordIdx < 0 || w1.equals(Glove.UNK))
                    continue;
                int windowStop = Math.min(i + windowSize + 1,tokens.size());
                for(int j = i; j < windowStop; j++) {
                    int otherWord = cache.indexOf(tokens.get(j));
                    String w2 = cache.wordFor(tokens.get(j)).getWord();
                    if(cache.indexOf(tokens.get(j)) < 0 || w2.equals(Glove.UNK))
                        continue;

                    if(otherWord == wordIdx)
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

