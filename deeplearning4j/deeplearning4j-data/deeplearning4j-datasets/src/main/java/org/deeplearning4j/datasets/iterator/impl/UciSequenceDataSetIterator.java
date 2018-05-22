package org.deeplearning4j.datasets.iterator.impl;

import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.fetchers.UciSequenceDataFetcher;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

/**
 * UCI synthetic control chart time series dataset. This dataset is useful for classification of univariate
 * time series with six categories:
 * Normal, Cyclic, Increasing trend, Decreasing trend, Upward shift, Downward shift
 *
 * Details:     https://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series
 * Data:        https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data
 * Image:       https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/data.jpeg
 *
 * @author Briton Park (bpark738)
 */
public class UciSequenceDataSetIterator extends SequenceRecordReaderDataSetIterator {

    protected DataSetPreProcessor preProcessor;

    public UciSequenceDataSetIterator(int batchSize) {
        this(batchSize, DataSetType.TRAIN, 123);
    }

    public UciSequenceDataSetIterator(int batchSize, DataSetType set) {
        this(batchSize, set, 123);
    }

    public UciSequenceDataSetIterator(int batchSize, DataSetType set, long rngSeed) {
        super(new UciSequenceDataFetcher().getRecordReader(rngSeed, set), batchSize, UciSequenceDataFetcher.NUM_LABELS, 1);
        // last parameter is index of label
    }
}