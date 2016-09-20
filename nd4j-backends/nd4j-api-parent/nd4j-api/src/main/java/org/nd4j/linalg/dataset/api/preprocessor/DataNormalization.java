package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;

/**
 * An interface for data normalizers.
 * Data normalizers compute some sort of statistics
 * over a dataset and scale the data in some way.
 *
 * @author Adam Gibson
 */
public interface DataNormalization extends org.nd4j.linalg.dataset.api.DataSetPreProcessor {
    /**
     * Fit a dataset (only compute
     * based on the statistics from this dataset0
     * @param dataSet the dataset to compute on
     */
    void fit(DataSet dataSet);

    /**
     * Iterates over a dataset
     * accumulating statistics for normalization
     * @param iterator the iterator to use for
     *                 collecting statistics.
     */
    void fit(DataSetIterator iterator);

    @Override
    void preProcess(DataSet toPreProcess);

    /**
     * Transform the dataset
     * @param toPreProcess the dataset to re process
     */
    void transform(DataSet toPreProcess);

    /**
     * Transform the dataset
     * @param features the features to pre process
     */
    void transform(INDArray features);

    /**
     * Load the statistics
     * for the data normalizer
     * @param statistics the files to persist
     * @throws IOException
     */
    void load(File...statistics) throws IOException;

    /**
     * Save the accumulated statistics
     * @param statistics the statistics to save
     * @throws IOException
     */
    void save(File...statistics) throws IOException;
}
