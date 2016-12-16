package org.nd4j.linalg.dataset.api.preprocessor;

import lombok.NonNull;
import org.nd4j.linalg.dataset.api.preprocessor.stats.NormalizerStats;

/**
 * Abstract base class for normalizers for both DataSet and MultiDataSet processing
 *
 * @author Ede Meijer
 */
abstract class AbstractNormalizer<S extends NormalizerStats> {
    protected NormalizerStrategy<S> strategy;

    protected AbstractNormalizer(@NonNull NormalizerStrategy<S> strategy) {
        this.strategy = strategy;
    }

    protected abstract boolean isFit();

    void assertIsFit() {
        if (!isFit()) {
            throw new RuntimeException(
                "API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)"
            );
        }
    }
}
