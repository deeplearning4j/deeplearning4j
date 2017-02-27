package org.nd4j.linalg.dataset.api.preprocessor;

import java.io.Serializable;

/**
 * Abstract base class for normalizers for both DataSet and MultiDataSet processing
 *
 * @author Ede Meijer
 */
public abstract class AbstractNormalizer implements Serializable {
    protected abstract boolean isFit();

    void assertIsFit() {
        if (!isFit()) {
            throw new RuntimeException(
                            "API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        }
    }
}
