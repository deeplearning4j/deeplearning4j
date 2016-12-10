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
     * Transform the labels. If {@link #isFitLabel()} == false, this is a no-op
     */
    void transformLabel(INDArray labels);

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance (arrays are modified in-place)
     *
     * @param toRevert    DataSet to revert the normalization on
     */
    void revert(DataSet toRevert);

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the specified features array
     *
     * @param features    Features to revert the normalization on
     */
    void revertFeatures(INDArray features);

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the specified labels array.
     * If labels normalization is disabled (i.e., {@link #isFitLabels()} == false) then this is a no-op.
     * Can also be used to undo normalization for network output arrays, in the case of regression.
     *
     * @param labels    Labels array to revert the normalization on
     */
    void revertLabels(INDArray labels);

    /**
     * Flag to specify if the labels/outputs in the dataset should be also normalized. Default value is usually false.
     */
    void fitLabel(boolean fitLabels);

    /**
     * Whether normalization for the labels is also enabled. Most commonly used for regression, not classification.
     *
     * @return True if labels will be
     */
    boolean isFitLabel();

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
