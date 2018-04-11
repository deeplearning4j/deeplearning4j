package org.deeplearning4j.datasets.iterator.tools;


import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Simple tool that generates pre-defined datastes
 * @author raver119@gmail.com
 */
public class MultiDataSetGenerator implements MultiDataSetIterator {

    protected final int[] shapeFeatures;
    protected final int[] shapeLabels;
    protected final long totalBatches;
    protected AtomicLong counter = new AtomicLong(0);

    public MultiDataSetGenerator(long numBatches, @NonNull int[] shapeFeatures, int[] shapeLabels) {
        this.shapeFeatures = shapeFeatures;
        this.shapeLabels = shapeLabels;
        this.totalBatches = numBatches;
    }

    public void shift() {
        counter.incrementAndGet();
    }

    @Override
    public MultiDataSet next(int num) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {

    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        counter.set(0);
    }

    @Override
    public boolean hasNext() {
        return counter.get() < totalBatches;
    }

    @Override
    public MultiDataSet next() {
        return new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[]{Nd4j.create(shapeFeatures).assign(counter.get())}, new INDArray[]{Nd4j.create(shapeLabels).assign(counter.getAndIncrement())});
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }
}
