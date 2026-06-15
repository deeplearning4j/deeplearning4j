package org.nd4j.linalg.dataset.curation.splitting;

import java.io.Serializable;
import java.util.List;

/**
 * Result of a train/validation/test split.
 */
public class SplitResult<T> implements Serializable {
    private static final long serialVersionUID = 1L;

    private final List<T> train;
    private final List<T> validation;
    private final List<T> test;

    public SplitResult(List<T> train, List<T> validation, List<T> test) {
        this.train = train;
        this.validation = validation;
        this.test = test;
    }

    public List<T> getTrain() {
        return train;
    }

    public List<T> getValidation() {
        return validation;
    }

    public List<T> getTest() {
        return test;
    }

    @Override
    public String toString() {
        return "SplitResult{train=" + train.size() + ", val=" + validation.size() + ", test=" + test.size() + "}";
    }
}
