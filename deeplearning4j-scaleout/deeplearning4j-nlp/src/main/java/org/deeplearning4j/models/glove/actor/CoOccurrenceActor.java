package org.deeplearning4j.models.glove.actor;

import akka.actor.UntypedActor;
import org.deeplearning4j.berkeley.CounterMap;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.movingwindow.Window;
import org.deeplearning4j.text.movingwindow.Windows;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

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
            List<Window> windows = Windows.windows(s,tokenizerFactory,windowSize);
            for(int i = 0; i < windows.size(); i++) {
                List<String> words = windows.get(i).getWords();
                String focusWord = windows.get(i).getFocusWord();
                int focusWordIdx = cache.indexOf(focusWord);
                if(focusWordIdx < 0)
                    continue;
                for(int j = 0; j < words.size(); j++) {
                    if(words.get(i).equals(focusWord)) {
                        continue;
                    }

                    int otherWordIdx = cache.indexOf(words.get(i));
                    if(otherWordIdx < focusWordIdx)
                        coOCurreneCounts.incrementCount(words.get(i),focusWord,1.0 / (j - i));
                    else
                        coOCurreneCounts.incrementCount(focusWord,words.get(i), 1.0 / (j - i));

                }
            }

            processed.incrementAndGet();

        }

        else
            unhandled(message);
    }
}

