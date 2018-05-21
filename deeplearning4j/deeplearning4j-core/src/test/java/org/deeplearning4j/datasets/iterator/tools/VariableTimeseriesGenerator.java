package org.deeplearning4j.datasets.iterator.tools;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This helper class generates
 * @author raver119@gmail.com
 */
@Slf4j
public class VariableTimeseriesGenerator implements DataSetIterator {
    protected Random rng;
    protected int batchSize;
    protected int values;
    protected int minTS, maxTS;
    protected int limit;
    protected int firstMaxima = 0;
    protected boolean isFirst = true;

    protected AtomicInteger counter = new AtomicInteger(0);

    public VariableTimeseriesGenerator(long seed, int numBatches, int batchSize, int values, int timestepsMin,
                    int timestepsMax) {
        this(seed, numBatches, batchSize, values, timestepsMin, timestepsMax, 0);
    }

    public VariableTimeseriesGenerator(long seed, int numBatches, int batchSize, int values, int timestepsMin,
                    int timestepsMax, int firstMaxima) {
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
    public DataSet next(int num) {
        int localMaxima = isFirst && firstMaxima > 0 ? firstMaxima
                        : minTS == maxTS ? minTS : rng.nextInt(maxTS - minTS) + minTS;

        if (isFirst)
            log.info("Local maxima: {}", localMaxima);

        isFirst = false;


        int[] shapeFeatures = new int[] {batchSize, values, localMaxima};
        int[] shapeLabels = new int[] {batchSize, 10};
        int[] shapeFMasks = new int[] {batchSize, localMaxima};
        int[] shapeLMasks = new int[] {batchSize, 10};
        //log.info("Allocating dataset seqnum: {}", counter.get());
        INDArray features = Nd4j.createUninitialized(shapeFeatures).assign(counter.get());
        INDArray labels = Nd4j.createUninitialized(shapeLabels).assign(counter.get() + 0.25);
        INDArray fMasks = Nd4j.createUninitialized(shapeFMasks).assign(counter.get() + 0.50);
        INDArray lMasks = Nd4j.createUninitialized(shapeLMasks).assign(counter.get() + 0.75);


        counter.getAndIncrement();

        return new DataSet(features, labels, fMasks, lMasks);
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        // no-op
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
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
        isFirst = true;
        counter.set(0);
    }

    @Override
    public boolean hasNext() {
        return counter.get() < limit;
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {

    }

    /**
     * Total examples in the iterator
     *
     * @return
     */
    @Override
    public int totalExamples() {
        return 0;
    }

    /**
     * Input columns for the dataset
     *
     * @return
     */
    @Override
    public int inputColumns() {
        return 0;
    }

    /**
     * The number of labels for the dataset
     *
     * @return
     */
    @Override
    public int totalOutcomes() {
        return 0;
    }

    /**
     * Batch size
     *
     * @return
     */
    @Override
    public int batch() {
        return 0;
    }

    /**
     * The current cursor if applicable
     *
     * @return
     */
    @Override
    public int cursor() {
        return 0;
    }

    /**
     * Total number of examples in the dataset
     *
     * @return
     */
    @Override
    public int numExamples() {
        return 0;
    }

    /**
     * Get dataset iterator record reader labels
     */
    @Override
    public List<String> getLabels() {
        return null;
    }
}
