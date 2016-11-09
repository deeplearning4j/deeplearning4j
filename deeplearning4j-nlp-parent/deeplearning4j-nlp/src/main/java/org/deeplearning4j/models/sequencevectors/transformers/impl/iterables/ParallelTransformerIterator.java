package org.deeplearning4j.models.sequencevectors.transformers.impl.iterables;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * TransformerIterator implementation that's does transformation/tokenization/normalization/whatever in parallel threads.
 * Suitable for cases when tokenization takes too much time for single thread.
 *
 * @author raver119@gmail.com
 */
public class ParallelTransformerIterator extends BasicTransformerIterator {

    protected Queue<Sequence<VocabWord>> buffer = new ConcurrentLinkedQueue<>();

    public ParallelTransformerIterator(@NonNull LabelAwareIterator iterator, @NonNull SentenceTransformer transformer) {
        this(iterator, transformer, true);
    }

    public ParallelTransformerIterator(@NonNull LabelAwareIterator iterator, @NonNull SentenceTransformer transformer, boolean allowMultithreading) {
        super(iterator, transformer);
        this.allowMultithreading = allowMultithreading;
    }

}
