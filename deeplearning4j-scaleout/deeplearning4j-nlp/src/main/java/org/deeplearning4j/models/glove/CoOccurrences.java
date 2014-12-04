package org.deeplearning4j.models.glove;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import akka.routing.RoundRobinPool;
import org.deeplearning4j.berkeley.CounterMap;
import org.deeplearning4j.models.glove.actor.CoOccurrenceActor;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;


import java.util.concurrent.atomic.AtomicInteger;

/**
 *
 * Co occurrence counts
 *
 * @author Adam Gibson
 */
public class CoOccurrences {
    private TokenizerFactory tokenizerFactory;
    private SentenceIterator sentenceIterator;
    private int windowSize = 15;
    protected transient VocabCache cache;
    protected InvertedIndex index;
    protected transient ActorSystem trainingSystem;

    private CounterMap<String,String> coOCurreneCounts = new CounterMap<>();

    public CoOccurrences(TokenizerFactory tokenizerFactory, SentenceIterator sentenceIterator, int windowSize, VocabCache cache, CounterMap<String, String> coOCurreneCounts) {
        this.tokenizerFactory = tokenizerFactory;
        this.sentenceIterator = sentenceIterator;
        this.windowSize = windowSize;
        this.cache = cache;
        this.coOCurreneCounts = coOCurreneCounts;
    }

    public void fit() {
        if(trainingSystem == null)
            trainingSystem = ActorSystem.create();

        final AtomicInteger processed = new AtomicInteger(0);

        final ActorRef actor = trainingSystem.actorOf(
                new RoundRobinPool(Runtime.getRuntime().availableProcessors()).props(
                        Props.create(
                                CoOccurrenceActor.class,
                                processed,
                                tokenizerFactory,
                                windowSize,
                                cache,
                                coOCurreneCounts)));


        sentenceIterator.reset();

        final AtomicInteger queued = new AtomicInteger(0);
        while(sentenceIterator.hasNext()) {
            String sentence = sentenceIterator.nextSentence();
            actor.tell(sentence,actor);
            queued.incrementAndGet();
        }


        while(processed.get() < queued.get()) {
            try {
                Thread.sleep(10000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }




    }

    public CounterMap<String, String> getCoOCurreneCounts() {
        return coOCurreneCounts;
    }

    public void setCoOCurreneCounts(CounterMap<String, String> coOCurreneCounts) {
        this.coOCurreneCounts = coOCurreneCounts;
    }

    public static class Builder {
        private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        private SentenceIterator sentenceIterator;
        private int windowSize = 15;
        private VocabCache cache;
        private CounterMap<String, String> coOCurreneCounts = new CounterMap<>();

        public Builder tokenizer(TokenizerFactory tokenizerFactory) {
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        public Builder iterate(SentenceIterator sentenceIterator) {
            this.sentenceIterator = sentenceIterator;
            return this;
        }

        public Builder windowSize(int windowSize) {
            this.windowSize = windowSize;
            return this;
        }

        public Builder cache(VocabCache cache) {
            this.cache = cache;
            return this;
        }

        public Builder coOCurreneCounts(CounterMap<String, String> coOCurreneCounts) {
            this.coOCurreneCounts = coOCurreneCounts;
            return this;
        }

        public CoOccurrences build() {
            if(cache == null)
                throw new IllegalArgumentException("Vocab cache must not be null!");

            if(sentenceIterator == null)
                throw new IllegalArgumentException("Sentence iterator must not be null");

            return new CoOccurrences(tokenizerFactory, sentenceIterator, windowSize, cache, coOCurreneCounts);
        }
    }




}
