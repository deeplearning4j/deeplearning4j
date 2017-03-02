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
public interface MultiDataNormalization extends Normalizer<MultiDataSet>, MultiDataSetPreProcessor {
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
}
