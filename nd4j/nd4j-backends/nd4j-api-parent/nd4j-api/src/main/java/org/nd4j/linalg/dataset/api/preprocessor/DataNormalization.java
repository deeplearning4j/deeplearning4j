package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * An interface for data normalizers.
 * Data normalizers compute some sort of statistics
 * over a dataset and scale the data in some way.
 *
 * @author Adam Gibson
 */
public interface DataNormalization extends Normalizer<DataSet>, DataSetPreProcessor {
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
     * @param features the features to pre process
     */
    void transform(INDArray features);

    /**
     * Transform the features, with an optional mask array
     * @param features the features to pre process
     * @param featuresMask the mask array to pre process
     */
    void transform(INDArray features, INDArray featuresMask);

    /**
     * Transform the labels. If {@link #isFitLabel()} == false, this is a no-op
     */
    void transformLabel(INDArray labels);

    /**
     * Transform the labels. If {@link #isFitLabel()} == false, this is a no-op
     */
    void transformLabel(INDArray labels, INDArray labelsMask);

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the specified features array
     *
     * @param features    Features to revert the normalization on
     */
    void revertFeatures(INDArray features);

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the specified features array
     *
     * @param features    Features to revert the normalization on
     * @param featuresMask
     */
    void revertFeatures(INDArray features, INDArray featuresMask);

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the specified labels array.
     * If labels normalization is disabled (i.e., {@link #isFitLabels()} == false) then this is a no-op.
     * Can also be used to undo normalization for network output arrays, in the case of regression.
     *
     * @param labels    Labels array to revert the normalization on
     */
    void revertLabels(INDArray labels);

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the specified labels array.
     * If labels normalization is disabled (i.e., {@link #isFitLabels()} == false) then this is a no-op.
     * Can also be used to undo normalization for network output arrays, in the case of regression.
     *
     * @param labels    Labels array to revert the normalization on
     * @param labelsMask Labels mask array (may be null)
     */
    void revertLabels(INDArray labels, INDArray labelsMask);

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
