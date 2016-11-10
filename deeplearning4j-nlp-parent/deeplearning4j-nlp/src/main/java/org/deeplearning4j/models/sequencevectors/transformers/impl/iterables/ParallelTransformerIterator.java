package org.deeplearning4j.models.sequencevectors.transformers.impl.iterables;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.transformers.SequenceTransformer;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.AsyncLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * TransformerIterator implementation that's does transformation/tokenization/normalization/whatever in parallel threads.
 * Suitable for cases when tokenization takes too much time for single thread.
 *
 * TL/DR: we read data from sentence iterator, and apply tokenization in parallel threads.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class ParallelTransformerIterator extends BasicTransformerIterator {

    protected LinkedBlockingQueue<Sequence<VocabWord>> buffer = new LinkedBlockingQueue<>(1024);
    protected LinkedBlockingQueue<LabelledDocument> stringBuffer;
    protected TokenizerThread[] threads = new TokenizerThread[Runtime.getRuntime().availableProcessors()];
    protected boolean underlyingHas = true;
    protected boolean exhausted = false;
    protected AtomicInteger processing = new AtomicInteger(0);

    public ParallelTransformerIterator(@NonNull LabelAwareIterator iterator, @NonNull SentenceTransformer transformer) {
        this(iterator, transformer, true);
    }

    public ParallelTransformerIterator(@NonNull LabelAwareIterator iterator, @NonNull SentenceTransformer transformer, boolean allowMultithreading) {
        super(new AsyncLabelAwareIterator(iterator, 256), transformer);
        this.allowMultithreading = allowMultithreading;
        this.stringBuffer = new LinkedBlockingQueue<>();

        try {
            int cnt = 0;
            while (this.iterator.hasNextDocument() && cnt < 32) {
                stringBuffer.put(this.iterator.nextDocument());
                cnt++;
            }
        } catch (InterruptedException e) {
            //
        }

        for (int x = 0; x < threads.length; x++) {
            threads[x] = new TokenizerThread(x, transformer,stringBuffer, buffer, processing);
            threads[x].start();
        }
    }

    @Override
    public boolean hasNext() {
        boolean before = underlyingHas;

        if (before)
            underlyingHas = iterator.hasNextDocument();

        return (underlyingHas || buffer.size() > 0 || stringBuffer.size() > 0 || processing.get() > 0);
    }

    @Override
    public Sequence<VocabWord> next() {
        try {
            if (underlyingHas)
                stringBuffer.put(iterator.nextDocument());

            return buffer.take();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }



    private static class TokenizerThread extends Thread implements Runnable {
        protected LinkedBlockingQueue<Sequence<VocabWord>> sequencesBuffer;
        protected LinkedBlockingQueue<LabelledDocument> stringsBuffer;
        protected SentenceTransformer sentenceTransformer;
        protected AtomicBoolean shouldWork = new AtomicBoolean(true);
        protected AtomicInteger processing;

        public TokenizerThread(int threadIdx, SentenceTransformer transformer, LinkedBlockingQueue<LabelledDocument> stringsBuffer, LinkedBlockingQueue<Sequence<VocabWord>> sequencesBuffer, AtomicInteger processing) {
            this.stringsBuffer = stringsBuffer;
            this.sequencesBuffer = sequencesBuffer;
            this.sentenceTransformer = transformer;
            this.processing = processing;

            this.setDaemon(true);
            this.setName("Tokenization thread " + threadIdx);
        }

        @Override
        public void run() {
            try {
                while (shouldWork.get()) {
                    LabelledDocument document = stringsBuffer.take();

                    processing.incrementAndGet();

                    Sequence<VocabWord> sequence = sentenceTransformer.transformToSequence(document.getContent());

                    if (sequence != null)
                        sequencesBuffer.put(sequence);

                    processing.decrementAndGet();
                }
            } catch (InterruptedException e) {
                // do nothing
                shutdown();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        public void shutdown() {
            shouldWork.set(false);
        }
    }
}
