package org.deeplearning4j.text.sentenceiterator;

import lombok.NonNull;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * This SentenceIterator implemenation wraps existing sentence iterator, and resets it numEpochs times
 *
 * This class is usable for tests purposes mostly.
 *
 * @author raver119@gmail.com
 */
public class MutipleEpochsSentenceIterator implements SentenceIterator {
    private SentenceIterator iterator;
    private int numEpochs;
    private AtomicInteger counter = new AtomicInteger(0);

    public MutipleEpochsSentenceIterator(@NonNull SentenceIterator iterator, int numEpochs) {
        this.numEpochs = numEpochs;
        this.iterator = iterator;

        this.iterator.reset();
    }

    @Override
    public String nextSentence() {
        return iterator.nextSentence();
    }

    @Override
    public boolean hasNext() {
        if (!iterator.hasNext()) {
            if (counter.get() < numEpochs - 1) {
                counter.incrementAndGet();
                iterator.reset();
                return true;
            } else
                return false;
        }
        return true;
    }

    @Override
    public void reset() {
        this.counter.set(0);
        this.iterator.reset();
    }

    @Override
    public void finish() {
        // no-op
    }

    @Override
    public SentencePreProcessor getPreProcessor() {
        return this.iterator.getPreProcessor();
    }

    @Override
    public void setPreProcessor(SentencePreProcessor preProcessor) {
        this.iterator.setPreProcessor(preProcessor);
    }
}
