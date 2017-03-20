package org.deeplearning4j.eval;

import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Not;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.LessThan;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * EvaluationBinary: used for evaluating networks with binary classification outputs. The typical classification metrics,
 * such as accuracy, precision, recall, F1 score, etc. are reported for each output.<br>
 * <p>
 * Note that EvaluationBinary supports both per-example and per-output masking.
 * <p>
 * The most common use case: multi-task networks, where each output is a binary value. This differs from {@link Evaluation}
 * in that {@link Evaluation} is for a single class (binary or non-binary) evaluation.
 *
 * @author Alex Black
 */
@NoArgsConstructor
public class EvaluationBinary extends BaseEvaluation<EvaluationBinary> {

    //Basically: Need counts - correct and incorrect - for each class
    //Because we want evaluation to work for large numbers of examples - and with low precision (FP16), we won't
    //use INDArrays to store the counts
    private int[] countTruePositive;    //P=1, Act=1
    private int[] countFalsePositive;   //P=1, Act=0
    private int[] countTrueNegative;    //P=0, Act=0
    private int[] countFalseNegative;   //P=0, Act=1

    private List<String> labels;

    public EvaluationBinary(int size) {
        countTruePositive = new int[size];
        countFalsePositive = new int[size];
        countTrueNegative = new int[size];
        countFalseNegative = new int[size];
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions) {
        eval(labels, networkPredictions, (INDArray) null);
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions, INDArray maskArray) {

        if (countTruePositive != null && countTruePositive.length != labels.size(1)) {
            throw new IllegalStateException("Labels array does not match stored state size. Expected labels array with "
                    + "size " + countTruePositive.length + ", got labels array with size " + labels.size(1));
        }

        //Assume 2d labels/predictions here.

        //First: binarize the network prediction probabilities, threshold 0.5
        //This gives us 3 binary arrays: labels, predictions, masks
        INDArray classPredictions = Nd4j.getExecutioner().execAndReturn(new LessThan(networkPredictions.dup()));

        INDArray notLabels = Nd4j.getExecutioner().execAndReturn(new Not(labels.dup()));
        INDArray notClassPredictions = Nd4j.getExecutioner().execAndReturn(new Not(classPredictions.dup()));

        INDArray truePositives = classPredictions.mul(labels);          //1s where predictions are 1, and labels are 1. 0s elsewhere
        INDArray trueNegatives = notClassPredictions.mul(notLabels);    //1s where predictions are 0, and labels are 0. 0s elsewhere
        INDArray falsePositives = classPredictions.mul(notLabels);      //1s where predictions are 1, labels are 0
        INDArray falseNegatives = notClassPredictions.mul(labels);      //1s where predictions are 0, labels are 1

        if (maskArray != null) {
            //By multiplying by mask, we keep only those 1s that are actually present
            truePositives.muli(maskArray);
            trueNegatives.muli(maskArray);
            falsePositives.muli(maskArray);
            falseNegatives.muli(maskArray);
        }

        int[] tpCount = truePositives.sum(0).data().asInt();
        int[] tnCount = trueNegatives.sum(0).data().asInt();
        int[] fpCount = falsePositives.sum(0).data().asInt();
        int[] fnCount = falseNegatives.sum(0).data().asInt();

        if (countTruePositive == null) {
            int l = tpCount.length;
            countTruePositive = new int[l];
            countFalsePositive = new int[l];
            countTrueNegative = new int[l];
            countFalsePositive = new int[l];
        }

        addInPlace(countTruePositive, tpCount);
        addInPlace(countFalsePositive, fpCount);
        addInPlace(countTrueNegative, tnCount);
        addInPlace(countFalseNegative, fnCount);
    }

    @Override
    public void merge(EvaluationBinary other) {
        if (other.countTruePositive == null) {
            //Other is empty - no op
            return;
        }

        if (countTruePositive == null) {
            //This evaluation is empty -> take results from other
            this.countTruePositive = other.countTruePositive;
            this.countFalsePositive = other.countFalsePositive;
            this.countTrueNegative = other.countTrueNegative;
            this.countFalseNegative = other.countFalseNegative;
        }
    }

    private static void addInPlace(int[] addTo, int[] toAdd) {
        for (int i = 0; i < addTo.length; i++) {
            addTo[i] += toAdd[i];
        }
    }

    /**
     * Returns the number of labels - (i.e., size of the prediction/labels arrays) - if the
     */
    public int numLabels() {
        if (countTruePositive == null) {
            return -1;
        }

        return countTruePositive.length;
    }

    /**
     * Set the label names
     */
    public void setLabelNames(List<String> labels) {
        if (labels == null) {
            this.labels = null;
            return;
        }
        this.labels = new ArrayList<>(labels);
    }

    public String stats() {
        StringBuilder sb = new StringBuilder();

        //Report: Accuracy, precision, recall, F1,
        //Then: confusion matrix

        int maxLabelsLength = 20;
        if (labels != null) {
            for (String s : labels) {
                maxLabelsLength = Math.max(s.length(), maxLabelsLength);
            }
        }

        int numDP = 5;
        String subPattern = "%-12." + numDP + "f";
        String pattern = "%-" + (maxLabelsLength + 5) + "s"             //Label
                + subPattern + subPattern + subPattern + subPattern     //Accuracy, precision, recall, f1
                + "%-10d-10d-10d-10d-10d";                              //Total count, TP, TN, FP, FN

        String patternHeader = "%-" + (maxLabelsLength+5) + "s"
                + "-12s-12s-12s-12s-10d-10d-10d-10d-10d";

        String header = String.format(patternHeader,
                "Label",
                "Accuracy",
                "F1",
                "Precision",
                "Recall",
                "Total #",
                "True Pos #",
                "True Neg #",
                "False Pos #",
                "False Neg #");

        sb.append(header);

        for (int i = 0; i < countTrueNegative.length; i++) {
            int tp = countTruePositive[i];
            int tn = countTrueNegative[i];
            int fp = countFalseNegative[i];
            int fn = countFalseNegative[i];
            int totalCount = tp + tn + fp + fn;

            double acc = (tp + tn) / (double) totalCount;
            double precision = tp / (double) (tp + fp);
            double recall = tp / (double) (tp + fn);
            double f1 = 2.0*(precision * recall) / (precision + recall);

            String label = (labels == null ? String.valueOf(i) : labels.get(i));

            sb.append("\n").append(String.format(pattern, acc, f1, precision, recall, totalCount, tp, tn, fp, fn));
        }

    }
}
