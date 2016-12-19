package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.stats.NormalizerStats;

import java.io.Serializable;

/**
 * Interface for strategies that can normalize and denormalize data arrays based on statistics of the population
 *
 * @author Ede Meijer
 */
interface NormalizerStrategy<S extends NormalizerStats> extends Serializable {
    /**
     * Normalize a data array
     *
     * @param array the data to normalize
     * @param stats statistics of the data population
     */
    void preProcess(INDArray array, INDArray maskArray, S stats);

    /**
     * Denormalize a data array
     *
     * @param array the data to denormalize
     * @param stats statistics of the data population
     */
    void revert(INDArray array, INDArray maskArray, S stats);
}
