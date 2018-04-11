package org.deeplearning4j.datasets.iterator.tools;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

public class DataSetGenerator implements DataSetIterator{
    protected final int[] shapeFeatures;
    protected final int[] shapeLabels;
    protected final long totalBatches;
    protected AtomicLong counter = new AtomicLong(0);

    public DataSetGenerator(long numBatches, @NonNull int[] shapeFeatures, int[] shapeLabels) {
        this.shapeFeatures = shapeFeatures;
        this.shapeLabels = shapeLabels;
        this.totalBatches = numBatches;
    }

    @Override
    public DataSet next(int i) {
        return null;
    }

    @Override
    public int totalExamples() {
        return 0;
    }

    @Override
    public int inputColumns() {
        return 0;
    }

    @Override
    public int totalOutcomes() {
        return 0;
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
    public int batch() {
        return 0;
    }

    @Override
    public int cursor() {
        return 0;
    }

    @Override
    public int numExamples() {
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {

    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return counter.get() < totalBatches;
    }

    @Override
    public DataSet next() {
        return new DataSet(Nd4j.create(shapeFeatures).assign(counter.get()), Nd4j.create(shapeLabels).assign(counter.getAndIncrement()));
    }

    @Override
    public void remove() {

    }

    public void shift() {
        counter.incrementAndGet();
    }
}
