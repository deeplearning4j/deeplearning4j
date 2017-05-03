package org.deeplearning4j.datasets.iterator.parallel;

import org.nd4j.linalg.dataset.api.iterator.enums.InequalityHandling;

import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseParallelDataSetIterator {
    protected AtomicLong counter = new AtomicLong(0);

    protected InequalityHandling inequalityHandling;




    public boolean hasNext() {
        // TODO: configurable probably?
        return true;
    }
}
