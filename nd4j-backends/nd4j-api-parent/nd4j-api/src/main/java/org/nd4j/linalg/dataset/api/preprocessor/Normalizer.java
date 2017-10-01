package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializerStrategy;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerType;

/**
 * Base interface for all normalizers
 *
 * @param <T> either {@link DataSet} or {@link MultiDataSet}
 */
public interface Normalizer<T> {
    /**
     * Fit a dataset (only compute based on the statistics from this dataset)
     *
     * @param dataSet the dataset to compute on
     */
    void fit(T dataSet);

    /**
     * Transform the dataset
     *
     * @param toPreProcess the dataset to re process
     */
    void transform(T toPreProcess);

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance (arrays are modified in-place)
     *
     * @param toRevert DataSet to revert the normalization on
     */
    void revert(T toRevert);

    /**
     * Get the enum opType of this normalizer
     *
     * @return the opType
     * @see NormalizerSerializerStrategy#getSupportedType()
     */
    NormalizerType getType();
}
