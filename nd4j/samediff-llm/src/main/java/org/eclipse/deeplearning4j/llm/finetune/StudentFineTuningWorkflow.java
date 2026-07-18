package org.eclipse.deeplearning4j.llm.finetune;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SimpleListMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.learning.config.IUpdater;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Configures and runs masked supervised fine-tuning on a caller-supplied causal LM graph.
 *
 * <p>Each batch must contain {@code features[0] = input_ids},
 * {@code features[1] = loss_mask}, and {@code labels[0] = labels}. The graph must expose
 * matching placeholders and already contain the masked causal-LM loss used for training.</p>
 */
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

    /** Evaluates the mean scalar loss over already materialized batches. */
    public double evaluateLoss(List<MultiDataSet> batches, String lossVariable) {
        if (batches == null || batches.isEmpty()) {
            throw new IllegalArgumentException("At least one batch is required");
        }
        if (lossVariable == null || lossVariable.trim().isEmpty()) {
            throw new IllegalArgumentException("lossVariable is required");
        }
        double total = 0.0;
        for (MultiDataSet batch : batches) {
            Map<String, INDArray> feed = new LinkedHashMap<>();
            feed.put(inputVariable, batch.getFeatures(0));
            feed.put(maskVariable, batch.getFeatures(1));
            feed.put(labelVariable, batch.getLabels(0));
            double loss = student.output(feed, lossVariable).get(lossVariable).getDouble(0);
            if (!Double.isFinite(loss)) {
                throw new IllegalStateException("Student loss is not finite: " + loss);
            }
            total += loss;
        }
        return total / batches.size();
    }
}
