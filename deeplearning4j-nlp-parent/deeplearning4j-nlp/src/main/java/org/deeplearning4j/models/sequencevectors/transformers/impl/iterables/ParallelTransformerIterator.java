package org.deeplearning4j.models.sequencevectors.transformers.impl.iterables;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.transformers.SequenceTransformer;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * TransformerIterator implementation that's does transformation/tokenization/normalization/whatever in parallel threads.
 * Suitable for cases when tokenization takes too much time for single thread.
 *
 * TL/DR: we read data from sentence iterator, and apply tokenization in parallel threads.
 *
 * @author raver119@gmail.com
 */
public class ParallelTransformerIterator extends BasicTransformerIterator {

    protected LinkedBlockingQueue<Sequence<VocabWord>> buffer = new LinkedBlockingQueue<>(1024);
    protected Queue<LabelledDocument> stringBuffer = new ConcurrentLinkedQueue<>();
    protected TokenizerThread[] threads = new TokenizerThread[6];

    public ParallelTransformerIterator(@NonNull LabelAwareIterator iterator, @NonNull SentenceTransformer transformer) {
        this(iterator, transformer, true);
    }

    public ParallelTransformerIterator(@NonNull LabelAwareIterator iterator, @NonNull SentenceTransformer transformer, boolean allowMultithreading) {
        super(iterator, transformer);
        this.allowMultithreading = allowMultithreading;

        for (int x = 0; x < threads.length; x++) {
            threads[x] = new TokenizerThread(x, transformer,stringBuffer, buffer);
            threads[x].start();
        }
    }

    @Override
    public boolean hasNext() {
        return buffer.size() > 0 || stringBuffer.size() > 0 || iterator.hasNextDocument();
    }

    @Override
    public Sequence<VocabWord> next() {
        try {
            int cnt = 0;
            while (cnt < 100 && stringBuffer.size() < 1000 && iterator.hasNextDocument()) {
                Object object = iterator.nextDocument();
                if (object != null && object instanceof LabelledDocument)
                    stringBuffer.add((LabelledDocument) object);
                cnt++;
            }

            Sequence<VocabWord> sequence = buffer.take();
            return sequence;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }



    private static class TokenizerThread extends Thread implements Runnable {
        protected LinkedBlockingQueue<Sequence<VocabWord>> sequencesBuffer;
        protected Queue<LabelledDocument> stringsBuffer;
        protected SentenceTransformer sentenceTransformer;
        protected AtomicBoolean shouldWork = new AtomicBoolean(true);

        public TokenizerThread(int threadIdx, SentenceTransformer transformer, Queue<LabelledDocument> stringsBuffer, LinkedBlockingQueue<Sequence<VocabWord>> sequencesBuffer) {
            this.stringsBuffer = stringsBuffer;
            this.sequencesBuffer = sequencesBuffer;
            this.sentenceTransformer = transformer;

            this.setDaemon(true);
            this.setName("Tokenization thread " + threadIdx);
        }

        @Override
        public void run() {
            while (shouldWork.get()) {
                LabelledDocument document = stringsBuffer.poll();

                if (document != null) {
                    Sequence<VocabWord> sequence = sentenceTransformer.transformToSequence(document.getContent());

                    if (sequence != null)
                        try {
                            sequencesBuffer.put(sequence);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                } else {
                    try {
                        Thread.sleep(100);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }
            }
        }

        public void shutdown() {
            shouldWork.set(false);
        }
    }
}
