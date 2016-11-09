package org.deeplearning4j.models.sequencevectors.transformers.impl.iterables;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * TransformerIterator implementation that's does transformation/tokenization/normalization/whatever in parallel threads.
 * Suitable for cases when tokenization takes too much time for single thread.
 *
 * TL/DR: we read data from sentence iterator, and apply tokenization in parallel threads.
 *
 * @author raver119@gmail.com
 */
public class ParallelTransformerIterator extends BasicTransformerIterator {

    protected Queue<Sequence<VocabWord>> buffer = new LinkedBlockingQueue<>(1024);
    protected Queue<String> stringBuffer = new ConcurrentLinkedQueue<>();

    public ParallelTransformerIterator(@NonNull LabelAwareIterator iterator, @NonNull SentenceTransformer transformer) {
        this(iterator, transformer, true);
    }

    public ParallelTransformerIterator(@NonNull LabelAwareIterator iterator, @NonNull SentenceTransformer transformer, boolean allowMultithreading) {
        super(iterator, transformer);
        this.allowMultithreading = allowMultithreading;
    }

    @Override
    public boolean hasNext() {
        return super.hasNext();
    }

    @Override
    public Sequence<VocabWord> next() {
        return super.next();
    }



    private static class TokenizerThread extends Thread implements Runnable {
        protected Queue<Sequence<VocabWord>> sequencesBuffer;
        protected Queue<String> stringsBuffer;

        public TokenizerThread(int threadIdx, Queue<String> stringsBuffer, Queue<Sequence<VocabWord>> sequencesBuffer) {
            this.stringsBuffer = stringsBuffer;
            this.sequencesBuffer = sequencesBuffer;

            this.setDaemon(true);
            this.setName("Tokenization thread " + threadIdx);
        }

        @Override
        public void run() {

        }
    }
}
