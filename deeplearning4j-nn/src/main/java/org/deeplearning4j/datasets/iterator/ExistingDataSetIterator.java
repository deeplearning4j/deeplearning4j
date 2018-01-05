package org.deeplearning4j.datasets.iterator;


import lombok.Getter;
import lombok.NonNull;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Iterator;
import java.util.List;

/**
 * This wrapper provides DataSetIterator interface to existing java Iterable<DataSet> and Iterator<DataSet>
 *
 * @author raver119@gmail.com
 */
public class ExistingDataSetIterator implements DataSetIterator {
    @Getter
    private DataSetPreProcessor preProcessor;

    private transient Iterable<DataSet> iterable;
    private transient Iterator<DataSet> iterator;
    private int totalExamples = 0;
    private int numFeatures = 0;
    private int numLabels = 0;
    private List<String> labels;


    public ExistingDataSetIterator(@NonNull Iterator<DataSet> iterator) {
        this.iterator = iterator;
    }

    public ExistingDataSetIterator(@NonNull Iterator<DataSet> iterator, @NonNull List<String> labels) {
        this(iterator);
        this.labels = labels;
    }

    public ExistingDataSetIterator(@NonNull Iterable<DataSet> iterable) {
        this.iterable = iterable;
        this.iterator = iterable.iterator();
    }

    public ExistingDataSetIterator(@NonNull Iterable<DataSet> iterable, @NonNull List<String> labels) {
        this(iterable);
        this.labels = labels;
    }


    public ExistingDataSetIterator(@NonNull Iterable<DataSet> iterable, int totalExamples, int numFeatures,
                    int numLabels) {
        this(iterable);

        this.totalExamples = totalExamples;
        this.numFeatures = numFeatures;
        this.numLabels = numLabels;
    }

    @Override
    public DataSet next(int num) {
        // TODO: this might be changed
        throw new UnsupportedOperationException("next(int) isn't supported");
    }

    @Override
    public int totalExamples() {
        return totalExamples;
    }

    @Override
    public int inputColumns() {
        return numFeatures;
    }

    @Override
    public int totalOutcomes() {
        if (labels != null)
            return labels.size();

        return numLabels;
    }

    @Override
    public boolean resetSupported() {
        return iterable != null;
    }

    @Override
    public boolean asyncSupported() {
        //No need to asynchronously prefetch here: already in memory
        return false;
    }

    @Override
    public void reset() {
        if (iterable != null)
            this.iterator = iterable.iterator();
        else
            throw new IllegalStateException(
                            "To use reset() method you need to provide Iterable<DataSet>, not Iterator");
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
        return totalExamples;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return labels;
    }

    @Override
    public boolean hasNext() {
        if (iterator != null)
            return iterator.hasNext();

        return false;
    }

    @Override
    public DataSet next() {
        if (preProcessor != null) {
            DataSet ds = iterator.next();
            if (!ds.isPreProcessed()) {
                preProcessor.preProcess(ds);
                ds.markAsPreProcessed();
            }
            return ds;
        } else
            return iterator.next();
    }

    @Override
    public void remove() {
        // no-op
    }
}
