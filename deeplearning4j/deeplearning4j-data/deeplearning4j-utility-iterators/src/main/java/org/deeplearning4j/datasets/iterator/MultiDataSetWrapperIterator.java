package org.deeplearning4j.datasets.iterator;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.List;

/**
 * This class is simple wrapper that takes single-input MultiDataSets and converts them to DataSets on the fly
 *
 * PLEASE NOTE: This only works if number of features/labels/masks is 1
 * @author raver119@gmail.com
 */
public class MultiDataSetWrapperIterator implements DataSetIterator {
    protected MultiDataSetIterator iterator;
    protected DataSetPreProcessor preProcessor;

    public MultiDataSetWrapperIterator(MultiDataSetIterator iterator) {
        this.iterator = iterator;
    }

    @Override
    public DataSet next(int num) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int inputColumns() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int totalOutcomes() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean resetSupported() {
        return iterator.resetSupported();
    }

    @Override
    public boolean asyncSupported() {
        return iterator.asyncSupported();
    }

    @Override
    public void reset() {
        iterator.reset();
    }

    @Override
    public int batch() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public DataSet next() {
        MultiDataSet mds = iterator.next();
        if (mds.getFeatures().length > 1 || mds.getLabels().length > 1)
            throw new UnsupportedOperationException(
                            "This iterator is able to convert MultiDataSet with number of inputs/outputs of 1");

        INDArray features = mds.getFeatures()[0];
        INDArray labels = mds.getLabels() != null ? mds.getLabels()[0] : features;
        INDArray fMask = mds.getFeaturesMaskArrays() != null ? mds.getFeaturesMaskArrays()[0] : null;
        INDArray lMask = mds.getLabelsMaskArrays() != null ? mds.getLabelsMaskArrays()[0] : null;

        DataSet ds = new DataSet(features, labels, fMask, lMask);

        if (preProcessor != null)
            preProcessor.preProcess(ds);

        return ds;
    }

    @Override
    public void remove() {
        // no-op
    }
}
