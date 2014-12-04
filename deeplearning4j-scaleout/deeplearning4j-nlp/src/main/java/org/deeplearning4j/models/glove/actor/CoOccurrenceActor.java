package org.deeplearning4j.models.glove.actor;

import akka.actor.UntypedActor;
import org.deeplearning4j.berkeley.CounterMap;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.movingwindow.Window;
import org.deeplearning4j.text.movingwindow.Windows;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Created by agibsonccc on 12/2/14.
 */
public class CoOccurrenceActor extends UntypedActor {
    private TokenizerFactory tokenizerFactory;
    private int windowSize = 5;
    private VocabCache cache;
    private CounterMap<String,String> coOCurreneCounts = new CounterMap<>();
    private AtomicInteger processed;
    public CoOccurrenceActor(AtomicInteger processed,TokenizerFactory tokenizerFactory, int windowSize, VocabCache cache, CounterMap<String, String> coOCurreneCounts) {
        this.processed = processed;
        this.tokenizerFactory = tokenizerFactory;
        this.windowSize = windowSize;
        this.cache = cache;
        this.coOCurreneCounts = coOCurreneCounts;
    }

    @Override
    public void onReceive(Object message) throws Exception {
        if(message instanceof String) {
            String s = (String) message;
            List<String> tokens = tokenizerFactory.create(s).getTokens();
            for(int i = 0; i < tokens.size(); i++) {
                int wordIdx = cache.indexOf(tokens.get(i));
                int windowStop = Math.min(i + windowSize + 1,tokens.size());
                for(int j = i; j < windowStop; j++) {
                    int otherWord = cache.indexOf(tokens.get(j));
                    if(otherWord == wordIdx)
                        continue;
                    if(otherWord < wordIdx)
                        coOCurreneCounts.incrementCount(tokens.get(j),tokens.get(i),1.0 / (j - i + Nd4j.EPS_THRESHOLD));
                    else
                        coOCurreneCounts.incrementCount(tokens.get(i),tokens.get(j), 1.0 / (j - i + Nd4j.EPS_THRESHOLD));


                }
            }

            processed.incrementAndGet();

        }

        else
            unhandled(message);
    }
}

