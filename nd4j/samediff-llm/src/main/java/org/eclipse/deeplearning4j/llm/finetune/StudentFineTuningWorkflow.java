package org.eclipse.deeplearning4j.llm.finetune;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.dataset.SimpleListMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.learning.config.IUpdater;

import java.util.List;

/** Configures and runs masked supervised fine-tuning on a caller-supplied causal LM graph. */
public class StudentFineTuningWorkflow {
    private final SameDiff student;
    private final IUpdater updater;
    private final String inputVariable;
    private final String maskVariable;
    private final String labelVariable;

    public StudentFineTuningWorkflow(SameDiff student, IUpdater updater) {
        this(student, updater, "input_ids", "loss_mask", "labels");
    }

    public StudentFineTuningWorkflow(SameDiff student, IUpdater updater,
                                     String inputVariable, String maskVariable, String labelVariable) {
        if (student == null) throw new IllegalArgumentException("student is required");
        if (updater == null) throw new IllegalArgumentException("updater is required");
        this.student = student;
        this.updater = updater;
        this.inputVariable = inputVariable;
        this.maskVariable = maskVariable;
        this.labelVariable = labelVariable;
    }

    public void train(List<MultiDataSet> batches, int epochs) {
        if (batches == null || batches.isEmpty()) throw new IllegalArgumentException("At least one batch is required");
        if (epochs < 1) throw new IllegalArgumentException("epochs must be positive");
        student.setTrainingConfig(TrainingConfig.builder()
                .updater(updater)
                .dataSetFeatureMapping(inputVariable, maskVariable)
                .dataSetLabelMapping(labelVariable)
                .build());
        student.prepareForTraining();
        student.fit(new SimpleListMultiDataSetIterator(batches), epochs);
    }
}
