package org.deeplearning4j.datasets.iterator.tools;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This helper class generates
 * @author raver119@gmail.com
 */
@Slf4j
public class VariableTimeseriesGenerator implements MultiDataSetIterator {
    protected Random rng;
    protected int batchSize;
    protected int values;
    protected int minTS, maxTS;
    protected int limit;
    protected int firstMaxima = 0;
    protected boolean isFirst = true;

    protected AtomicInteger counter = new AtomicInteger(0);

    public VariableTimeseriesGenerator(long seed, int numBatches, int batchSize, int values, int timestepsMin, int timestepsMax) {
        this(seed, numBatches, batchSize, values, timestepsMin, timestepsMax, 0);
    }

    public VariableTimeseriesGenerator(long seed, int numBatches, int batchSize, int values, int timestepsMin, int timestepsMax, int firstMaxima) {
        this.rng = new Random(seed);
        this.values = values;
        this.batchSize = batchSize;
        this.limit = numBatches;
        this.maxTS = timestepsMax;
        this.minTS = timestepsMin;
        this.firstMaxima = firstMaxima;

        if (timestepsMax < timestepsMin)
            throw new DL4JInvalidConfigException("timestepsMin should be <= timestepsMax");
    }


    @Override
    public MultiDataSet next(int num) {
        int localMaxima = isFirst && firstMaxima > 0 ? firstMaxima: rng.nextInt(maxTS - minTS) + minTS;
        log.info("Local maxima: {}", localMaxima);

        isFirst = false;

        int[] shapeFeatures = new int[]{batchSize, values, localMaxima};
        int[] shapeLabels = new int[] {batchSize, 10};
        int[] shapeFMasks = new int[] {batchSize, localMaxima};
        int[] shapeLMasks = new int[] {batchSize, 10};
        INDArray features = Nd4j.create(shapeFeatures).assign(counter.get());
        INDArray labels = Nd4j.create(shapeLabels).assign(counter.get() + 0.25);
        INDArray fMasks = Nd4j.create(shapeFMasks).assign(counter.get() + 0.50);
        INDArray lMasks = Nd4j.create(shapeLMasks).assign(counter.get() + 0.75);


        counter.getAndIncrement();

        return new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[]{features}, new INDArray[]{labels}, new INDArray[]{fMasks}, new INDArray[]{lMasks});
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        // no-op
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
        return counter.get() < limit;
    }

    @Override
    public MultiDataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {

    }
}
