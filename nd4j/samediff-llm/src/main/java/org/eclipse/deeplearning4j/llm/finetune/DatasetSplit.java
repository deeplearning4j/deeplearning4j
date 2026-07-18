package org.eclipse.deeplearning4j.llm.finetune;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Immutable train, validation, and test partitions. */
public final class DatasetSplit {
    private final List<GeneratedTrainingExample> train;
    private final List<GeneratedTrainingExample> validation;
    private final List<GeneratedTrainingExample> test;

    DatasetSplit(List<GeneratedTrainingExample> train, List<GeneratedTrainingExample> validation,
                 List<GeneratedTrainingExample> test) {
        this.train = Collections.unmodifiableList(new ArrayList<>(train));
        this.validation = Collections.unmodifiableList(new ArrayList<>(validation));
        this.test = Collections.unmodifiableList(new ArrayList<>(test));
    }

    public List<GeneratedTrainingExample> getTrain() { return train; }
    public List<GeneratedTrainingExample> getValidation() { return validation; }
    public List<GeneratedTrainingExample> getTest() { return test; }
}
