package org.deeplearning4j.models.glove;

import lombok.NonNull;
import org.deeplearning4j.models.abstractvectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.abstractvectors.iterators.SynchronizedSequenceIterator;
import org.deeplearning4j.models.abstractvectors.sequence.Sequence;
import org.deeplearning4j.models.abstractvectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * @author raver119@gmail.com
 */
public class AbstractCoOccurrences<T extends SequenceElement> {

    protected boolean symmetric;
    protected int windowSize;
    protected VocabCache<T> vocabCache;
    protected SequenceIterator<T> sequenceIterator;
    protected int workers = Runtime.getRuntime().availableProcessors();


    public static class Builder<T extends SequenceElement> {

        protected boolean symmetric;
        protected int windowSize = 5;
        protected VocabCache<T> vocabCache;
        protected SequenceIterator<T> sequenceIterator;
        protected int workers = Runtime.getRuntime().availableProcessors();


        public Builder() {

        }

        public Builder<T> symmetric(boolean reallySymmetric) {
            this.symmetric = reallySymmetric;
            return this;
        }

        public Builder<T> windowSize(int windowSize) {
            this.windowSize = windowSize;
            return this;
        }

        public Builder<T> vocabCache(@NonNull VocabCache<T> cache) {
            this.vocabCache = cache;
            return this;
        }

        public Builder<T> iterate(@NonNull SequenceIterator<T> iterator) {
            this.sequenceIterator = new SynchronizedSequenceIterator<T>(iterator);
            return this;
        }

        public Builder<T> workers(int numWorkers) {
            this.workers = numWorkers;
            return this;
        }

        public AbstractCoOccurrences<T> build() {
            AbstractCoOccurrences<T> ret = new AbstractCoOccurrences<>();
            ret.sequenceIterator = this.sequenceIterator;
            ret.windowSize = this.windowSize;
            ret.vocabCache = this.vocabCache;
            ret.symmetric = this.symmetric;
            ret.workers = this.workers;

            return ret;
        }
    }

    private class CoOccurrencesCalculatorThread<T extends SequenceElement> extends Thread implements Runnable {

        private final SequenceIterator<T> iterator;

        public CoOccurrencesCalculatorThread(int threadId, @NonNull SequenceIterator<T> iterator) {
            this.iterator = iterator;
            this.setName("CoOccurrencesCalculatorThread " + threadId);
        }

        @Override
        public void run() {
            while (iterator.hasMoreSequences()) {
                Sequence<T> sequence = iterator.nextSequence();

                // TODO: vocab filtering should take place

                List<String> tokens = new ArrayList<>(vocabCache.words());
                for (int x = 0; x < sequence.getElements().size(); x++) {
                    int wordIdx = vocabCache.indexOf(tokens.get(x));
                    if (wordIdx < 0) continue;
                    String w1 = vocabCache.wordFor(tokens.get(i)).getLabel();

                    if(w1.equals(Glove.UNK))
                        continue;
                    int windowStop = Math.min(x + windowSize + 1,tokens.size());
                    for(int j = x; j < windowStop; j++) {
                        int otherWord = tokens.indexOf(tokens.get(j));
                        if (otherWord < 0) continue;
                        String w2 = vocabCache.wordFor(tokens.get(j)).getLabel();
                        if(w2.equals(Glove.UNK) || otherWord == wordIdx)
                            continue;
                        if(wordIdx < otherWord) {
                            coOCurreneCounts.incrementCount(tokens.get(x), tokens.get(j), 1.0 / (j - i + Nd4j.EPS_THRESHOLD));
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
        }
    }
}