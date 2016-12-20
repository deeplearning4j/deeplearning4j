package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * An interface for multi dataset normalizers.
 * Data normalizers compute some sort of statistics
 * over a MultiDataSet and scale the data in some way.
 *
 * @author Ede Meijer
 */
public interface MultiDataNormalization extends MultiDataSetPreProcessor {
    /**
     * Fit a multi dataset (only compute based on the statistics from this dataset)
     *
     * @param dataSet the dataset to compute on
     */
    void fit(MultiDataSet dataSet);

    /**
     * Iterates over a dataset
     * accumulating statistics for normalization
     *
     * @param iterator the iterator to use for
     *                 collecting statistics.
     */
    void fit(MultiDataSetIterator iterator);

    @Override
    void preProcess(MultiDataSet multiDataSet);

    /**
     * Transform the dataset
     *
     * @param toPreProcess the dataset to re process
     */
    void transform(MultiDataSet toPreProcess);

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance (arrays are modified in-place)
     *
     * @param toRevert DataSet to revert the normalization on
     */
    void revert(MultiDataSet toRevert);

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the specified features array
     *
     * @param features    Features to revert the normalization on
     * @param featuresMask
     */
    void revertFeatures(INDArray[] features, INDArray[] featuresMask);

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the specified features array
     *
     * @param features Features to revert the normalization on
     */
    void revertFeatures(INDArray[] features);

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the specified labels array.
     * If labels normalization is disabled (i.e., {@link #isFitLabel()} == false) then this is a no-op.
     * Can also be used to undo normalization for network output arrays, in the case of regression.
     *
     * @param labels    Labels array to revert the normalization on
     * @param labelsMask Labels mask array (may be null)
     */
    void revertLabels(INDArray[] labels, INDArray[] labelsMask);

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the specified labels array.
     * If labels normalization is disabled (i.e., {@link #isFitLabel()} == false) then this is a no-op.
     * Can also be used to undo normalization for network output arrays, in the case of regression.
     *
     * @param labels Labels array to revert the normalization on
     */
    void revertLabels(INDArray[] labels);

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
}
