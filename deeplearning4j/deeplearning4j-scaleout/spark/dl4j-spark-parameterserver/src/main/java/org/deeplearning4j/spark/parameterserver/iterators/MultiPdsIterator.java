package org.deeplearning4j.spark.parameterserver.iterators;

import lombok.NonNull;
import org.apache.spark.input.PortableDataStream;
import org.deeplearning4j.spark.parameterserver.callbacks.PortableDataStreamMDSCallback;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.util.Iterator;
import java.util.function.Consumer;

/**
 * @author raver119@gmail.com
 */
public class MultiPdsIterator implements Iterator<MultiDataSet> {
    protected final Iterator<PortableDataStream> iterator;
    protected final PortableDataStreamMDSCallback callback;

    public MultiPdsIterator(@NonNull Iterator<PortableDataStream> pds,
                    @NonNull PortableDataStreamMDSCallback callback) {
        this.iterator = pds;
        this.callback = callback;
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public MultiDataSet next() {
        return callback.compute(iterator.next());
    }

    @Override
    public void remove() {
        // no-op
    }

    @Override
    public void forEachRemaining(Consumer<? super MultiDataSet> action) {
        throw new UnsupportedOperationException();
    }
}
