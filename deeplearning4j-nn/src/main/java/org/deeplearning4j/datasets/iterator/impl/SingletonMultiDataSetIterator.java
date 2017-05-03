package org.deeplearning4j.datasets.iterator.impl;

import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.NoSuchElementException;

/**
 * A very simple adapter class for converting a single MultiDataSet to a MultiDataSetIterator.
 * Returns a single MultiDataSet
 *
 * @author Alex Black
 */
public class SingletonMultiDataSetIterator implements MultiDataSetIterator {

    private final MultiDataSet multiDataSet;
    private boolean hasNext = true;
    private boolean preprocessed = false;
    private MultiDataSetPreProcessor preProcessor;

    public SingletonMultiDataSetIterator(MultiDataSet multiDataSet) {
        this.multiDataSet = multiDataSet;
    }

    @Override
    public MultiDataSet next(int num) {
        return next();
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        hasNext = true;
    }

    @Override
    public boolean hasNext() {
        return hasNext;
    }

    @Override
    public MultiDataSet next() {
        if (!hasNext) {
            throw new NoSuchElementException("No elements remaining");
        }
        hasNext = false;
        if (preProcessor != null && !preprocessed) {
            preProcessor.preProcess(multiDataSet);
            preprocessed = true;
        }
        return multiDataSet;
    }

    @Override
    public void remove() {
        //No op
    }
}
