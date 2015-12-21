package org.deeplearning4j.models.glove;

import lombok.NonNull;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.CounterMap;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.iterators.FilteredSequenceIterator;
import org.deeplearning4j.models.sequencevectors.iterators.SynchronizedSequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.movingwindow.Util;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * @author raver119@gmail.com
 */
public class AbstractCoOccurrences<T extends SequenceElement> implements Serializable {

    protected boolean symmetric;
    protected int windowSize;
    protected VocabCache<T> vocabCache;
    protected SequenceIterator<T> sequenceIterator;
    protected int workers = Runtime.getRuntime().availableProcessors();

    private Counter<Integer> sentenceOccurrences = Util.parallelCounter();
    private CounterMap<T, T> coOCurreneCounts = Util.parallelCounterMap();
    private Counter<Integer> occurrenceAllocations = Util.parallelCounter();
    private List<Pair<T, T>> coOccurrences;
    private AtomicLong processedSequences = new AtomicLong(0);

    protected static final Logger logger = LoggerFactory.getLogger(AbstractCoOccurrences.class);

    public double getCoOccurrenceCount(@NonNull T element1, @NonNull T element2) {
        return coOCurreneCounts.getCount(element1, element2);
    }

    public void fit() {

        sequenceIterator.reset();

        List<CoOccurrencesCalculatorThread> threads = new ArrayList<>();
        for (int x = 0; x < workers; x++) {
            threads.add(x, new CoOccurrencesCalculatorThread<T>(x, new FilteredSequenceIterator<T>(new SynchronizedSequenceIterator<T>(sequenceIterator), vocabCache), processedSequences));
            threads.get(x).start();
        }

        for (int x = 0; x < workers; x++) {
            try {
                threads.get(x).join();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        logger.info("CoOccurrences map was built: ["+ coOCurreneCounts.size()+"]");
    }

    /**
     * Returns list of label pairs for each element met in each sequence
     * @return
     */
    public synchronized List<Pair<T, T>> coOccurrenceList() {
        if (coOccurrences != null)
            return coOccurrences;

        coOccurrences = new ArrayList<>();
        Iterator<Pair<T, T>> iterator = coOCurreneCounts.getPairIterator();
        while (iterator.hasNext()) {
            Pair<T, T> pair = iterator.next();

            if (pair.getFirst().equals(pair.getSecond())) continue;

            // each pair should be checked against vocab, but that's not strictly required
            if (!vocabCache.hasToken(pair.getFirst().getLabel()) || !vocabCache.hasToken(pair.getSecond().getLabel())) {
//                logger.debug("Skipping pair: '"+ pair.getFirst()+"', '"+ pair.getSecond()+"'");
                continue;
            }// else logger.debug("Adding pair: '"+ pair.getFirst()+"', '"+ pair.getSecond()+"'");



            coOccurrences.add(new Pair<T, T>(pair.getFirst(), pair.getSecond()));
            if (coOccurrences.size() % 100000 == 0) logger.info("Cooccurrences gathered: " + coOccurrences.size());
        }

        return coOccurrences;
    }

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
        private final AtomicLong sequenceCounter;

        public CoOccurrencesCalculatorThread(int threadId, @NonNull SequenceIterator<T> iterator, @NonNull AtomicLong sequenceCounter) {
            this.iterator = iterator;
            this.sequenceCounter = sequenceCounter;
            this.setName("CoOccurrencesCalculatorThread " + threadId);
        }

        @Override
        public void run() {
            while (iterator.hasMoreSequences()) {
                Sequence<T> sequence = iterator.nextSequence();

//                logger.info("Sequence ID: " + sequence.getSequenceId());
                // TODO: vocab filtering should take place

                List<String> tokens = new ArrayList<>(sequence.asLabels());
    //            logger.info("Tokens size: " + tokens.size());
                for (int x = 0; x < sequence.getElements().size(); x++) {
                    int wordIdx = vocabCache.indexOf(tokens.get(x));
                    if (wordIdx < 0) continue;
                    String w1 = vocabCache.wordFor(tokens.get(x)).getLabel();

                    // THIS iS SAFE TO REMOVE, NO CHANCE WE'll HAVE UNK WORD INSIDE SEQUENCE
                    /*if(w1.equals(Glove.UNK))
                        continue;
                    */

                    int windowStop = Math.min(x + windowSize + 1,tokens.size());
                    for(int j = x; j < windowStop; j++) {
                        int otherWord = vocabCache.indexOf(tokens.get(j));
                        if (otherWord < 0) continue;
                        String w2 = vocabCache.wordFor(tokens.get(j)).getLabel();

                        if(w2.equals(Glove.UNK) || otherWord == wordIdx) {
                            continue;
                        }


                        if(wordIdx < otherWord) {
                            coOCurreneCounts.incrementCount(vocabCache.wordFor(tokens.get(x)), vocabCache.wordFor(tokens.get(j)), 1.0 / (j - x + Nd4j.EPS_THRESHOLD));
                            occurrenceAllocations.incrementCount(sequence.getSequenceId(),1.0);
                            if(symmetric) {
                                coOCurreneCounts.incrementCount(vocabCache.wordFor(tokens.get(j)), vocabCache.wordFor(tokens.get(x)), 1.0 / (j - x + Nd4j.EPS_THRESHOLD));
                                occurrenceAllocations.incrementCount(sequence.getSequenceId(),1.0);
                            }
                        }
                        else {
                            coOCurreneCounts.incrementCount(vocabCache.wordFor(tokens.get(j)),vocabCache.wordFor(tokens.get(x)), 1.0 / (j - x + Nd4j.EPS_THRESHOLD));
                            occurrenceAllocations.incrementCount(sequence.getSequenceId(),1.0);

                            if(symmetric) {
                                coOCurreneCounts.incrementCount(vocabCache.wordFor(tokens.get(x)), vocabCache.wordFor(tokens.get(j)), 1.0 / (j - x + Nd4j.EPS_THRESHOLD));
                                occurrenceAllocations.incrementCount(sequence.getSequenceId(),1.0);
                            }
                        }
                    }
                }

                sequenceCounter.incrementAndGet();
            }
        }
    }
}