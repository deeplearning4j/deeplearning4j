package org.nd4j.linalg.dataset.api.iterator;

import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Ede Meijer
 */
public class TestMultiDataSetIterator implements MultiDataSetIterator {
    private int curr = 0;
    private int batch = 10;
    private List<MultiDataSet> list;
    private MultiDataSetPreProcessor preProcessor;

    /**
     * Makes an iterator from the given dataset
     * ONLY for use in tests in nd4j
     * Initializes with a default batch of 5
     */
    public TestMultiDataSetIterator(MultiDataSet dataset) {
        this(dataset, 5);

    }

    public TestMultiDataSetIterator(MultiDataSet dataset, int batch) {
        list = new ArrayList<>(dataset.asList());
        this.batch = batch;
    }

    @Override
    public MultiDataSet next(int num) {
        int end = curr + num;

        List<MultiDataSet> r = new ArrayList<>();
        if (end >= list.size()) {
            end = list.size();
        }
        for (; curr < end; curr++) {
            r.add(list.get(curr));
        }

        MultiDataSet d = org.nd4j.linalg.dataset.MultiDataSet.merge(r);
        if (preProcessor != null) {
            preProcessor.preProcess(d);
        }
        return d;
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
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
        curr = 0;
    }

    @Override
    public boolean hasNext() {
        return curr < list.size();
    }

    @Override
    public MultiDataSet next() {
        return next(batch);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }
}
