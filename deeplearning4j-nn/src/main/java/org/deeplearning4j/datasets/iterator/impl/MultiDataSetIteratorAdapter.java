package org.deeplearning4j.datasets.iterator.impl;

import org.deeplearning4j.nn.graph.util.ComputationGraphUtil;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * Iterator that adapts a DataSetIterator to a MultiDataSetIterator
 *
 * @author Alex Black
 */
public class MultiDataSetIteratorAdapter implements MultiDataSetIterator {

    private org.nd4j.linalg.dataset.api.iterator.DataSetIterator iter;
    private MultiDataSetPreProcessor preProcessor;

    public MultiDataSetIteratorAdapter(org.nd4j.linalg.dataset.api.iterator.DataSetIterator iter) {
        this.iter = iter;
    }

    @Override
    public MultiDataSet next(int i) {
        MultiDataSet mds = ComputationGraphUtil.toMultiDataSet(iter.next(i));
        if (preProcessor != null)
            preProcessor.preProcess(mds);
        return mds;
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor multiDataSetPreProcessor) {
        this.preProcessor = multiDataSetPreProcessor;
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public boolean resetSupported() {
        return iter.resetSupported();
    }

    @Override
    public boolean asyncSupported() {
        return iter.asyncSupported();
    }

    @Override
    public void reset() {
        iter.reset();
    }

    @Override
    public boolean hasNext() {
        return iter.hasNext();
    }

    @Override
    public MultiDataSet next() {
        MultiDataSet mds = ComputationGraphUtil.toMultiDataSet(iter.next());
        if (preProcessor != null)
            preProcessor.preProcess(mds);
        return mds;
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

}