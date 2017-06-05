package org.deeplearning4j.spark.parameterserver.iterators;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.ParallelDataSetIterator;

import java.util.List;

/**
 * This DataSetIterator implementation does accumulation of DataSets from different Spark executors, wrt Thread/Device Affinity
 *
 * @author raver119@gmail.com
 */
public class VirtualDataSetIterator implements ParallelDataSetIterator {

    @Override
    public void attachThread(int producer) {

    }

    @Override
    public boolean hasNextFor() {
        return false;
    }

    @Override
    public boolean hasNextFor(int consumer) {
        return false;
    }

    @Override
    public DataSet nextFor(int consumer) {
        return null;
    }

    @Override
    public DataSet nextFor() {
        return null;
    }

    @Override
    public DataSet next(int num) {
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
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {

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
    public void setPreProcessor(DataSetPreProcessor preProcessor) {

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
        return false;
    }

    @Override
    public DataSet next() {
        return null;
    }

    @Override
    public void remove() {

    }
}
