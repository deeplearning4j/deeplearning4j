package org.deeplearning4j.text.documentiterator;

import lombok.Getter;
import lombok.NonNull;
import org.deeplearning4j.parallelism.AsyncIterator;

import java.util.Iterator;

/**
 * @author raver119@gmail.com
 */
public class AsyncLabelAwareIterator implements LabelAwareIterator, Iterator<LabelledDocument> {

    protected LabelAwareIterator backedIterator;
    @Getter
    protected AsyncIterator<LabelledDocument> asyncIterator;
    protected int bufferSize;

    public AsyncLabelAwareIterator(@NonNull LabelAwareIterator iterator, int bufferSize) {
        this.backedIterator = iterator;
        this.bufferSize = bufferSize;
        this.asyncIterator = new AsyncIterator<>(backedIterator, bufferSize);
    }

    @Override
    public void remove() {
        // no-op
    }

    @Override
    public boolean hasNextDocument() {
        return asyncIterator.hasNext();
    }

    @Override
    public LabelledDocument nextDocument() {
        return asyncIterator.next();
    }

    @Override
    public void reset() {
        asyncIterator.shutdown();
        backedIterator.reset();
        asyncIterator = new AsyncIterator<>(backedIterator, bufferSize);
    }

    @Override
    public void shutdown() {
        asyncIterator.shutdown();
        backedIterator.shutdown();
    }

    @Override
    public LabelsSource getLabelsSource() {
        return backedIterator.getLabelsSource();
    }

    @Override
    public boolean hasNext() {
        return hasNextDocument();
    }

    @Override
    public LabelledDocument next() {
        return nextDocument();
    }
}
