package org.deeplearning4j.datasets.iterator.impl;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.atomic.AtomicLong;

/**
 * MultiDataset iterator for simulated inputs, or input derived from a MultiDataSet example. Primarily
 * used for benchmarking.
 *
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public class BenchmarkMultiDataSetIterator implements MultiDataSetIterator {
    private INDArray[] baseFeatures;
    private INDArray[] baseLabels;
    private long limit;
    private AtomicLong counter = new AtomicLong(0);

    public BenchmarkMultiDataSetIterator(int[][] featuresShape, int[] numLabels, int totalIterations) {
        if (featuresShape.length != numLabels.length)
            throw new IllegalArgumentException("Number of input features must match length of input labels.");

        this.baseFeatures = new INDArray[featuresShape.length];
        for (int i = 0; i < featuresShape.length; i++) {
            baseFeatures[i] = Nd4j.rand(featuresShape[i]);
        }
        this.baseLabels = new INDArray[featuresShape.length];
        for (int i = 0; i < featuresShape.length; i++) {
            baseLabels[i] = Nd4j.create(featuresShape[i][0], numLabels[i]);
            baseLabels[i].getColumn(1).assign(1.0);
        }

        Nd4j.getExecutioner().commit();
        this.limit = totalIterations;
    }

    public BenchmarkMultiDataSetIterator(MultiDataSet example, int totalIterations) {
        this.baseFeatures = new INDArray[example.getFeatures().length];
        for (int i = 0; i < example.getFeatures().length; i++) {
            baseFeatures[i] = example.getFeatures()[i].dup();
        }
        this.baseLabels = new INDArray[example.getLabels().length];
        for (int i = 0; i < example.getLabels().length; i++) {
            baseFeatures[i] = example.getLabels()[i].dup();
        }

        Nd4j.getExecutioner().commit();
        this.limit = totalIterations;
    }

    @Override
    public MultiDataSet next(int i) {
        throw new UnsupportedOperationException();
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
        this.counter.set(0);
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor dataSetPreProcessor) {

    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return null;
    }

    /**
     * Returns {@code true} if the iteration has more elements.
     * (In other words, returns {@code true} if {@link #next} would
     * return an element rather than throwing an exception.)
     *
     * @return {@code true} if the iteration has more elements
     */
    @Override
    public boolean hasNext() {
        return counter.get() < limit;
    }

    /**
     * Returns the next element in the iteration.
     *
     * @return the next element in the iteration
     */
    @Override
    public MultiDataSet next() {
        counter.incrementAndGet();

        INDArray[] features = new INDArray[baseFeatures.length];
        for (int i = 0; i < baseFeatures.length; i++) {
            features[i] = baseFeatures[i];
        }
        INDArray[] labels = new INDArray[baseLabels.length];
        for (int i = 0; i < baseLabels.length; i++) {
            labels[i] = baseLabels[i];
        }

        MultiDataSet ds = new MultiDataSet(features, labels);

        return ds;
    }

    /**
     * Removes from the underlying collection the last element returned
     * by this iterator (optional operation).  This method can be called
     * only once per call to {@link #next}.  The behavior of an iterator
     * is unspecified if the underlying collection is modified while the
     * iteration is in progress in any way other than by calling this
     * method.
     *
     * @throws UnsupportedOperationException if the {@code remove}
     *                                       operation is not supported by this iterator
     * @throws IllegalStateException         if the {@code next} method has not
     *                                       yet been called, or the {@code remove} method has already
     *                                       been called after the last call to the {@code next}
     *                                       method
     * @implSpec The default implementation throws an instance of
     * {@link UnsupportedOperationException} and performs no other action.
     */
    @Override
    public void remove() {

    }
}
