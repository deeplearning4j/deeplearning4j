package org.deeplearning4j.eval;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.io.Serializable;
import java.util.*;

/**
 * Created by Alex on 04/11/2016.
 */
public class ROC {

    private final int thresholdSteps;

    private long countActualPositive;
    private long countActualNegative;

    private final Map<Double, CountsForThreshold> counts = new LinkedHashMap<>();

    public ROC(int thresholdSteps) {
        this.thresholdSteps = thresholdSteps;

        double step = 1.0 / thresholdSteps;
        for (int i = 0; i <= thresholdSteps; i++) {
            double currThreshold = i * step;
            counts.put(currThreshold, new CountsForThreshold(currThreshold));
        }
    }


    public void eval(INDArray outcomes, INDArray predictions) {

        double step = 1.0 / thresholdSteps;
        boolean singleOutput = outcomes.size(1) == 1;

        //For now: assume 2d data. Each row: has 2 values (TODO: single binary variable case)
//        INDArray positivePredictedClassColumn = predictions.getColumn(1);
//        INDArray positiveActualClassColumn = outcomes.getColumn(1);
//        INDArray negativeActualClassColumn = outcomes.getColumn(0);
        INDArray positivePredictedClassColumn;
        INDArray positiveActualClassColumn;
        INDArray negativeActualClassColumn;

        if(singleOutput){
            //Single binary variable case
            positiveActualClassColumn = outcomes;
            negativeActualClassColumn = outcomes.rsub(1.0); //1.0 - label
            positivePredictedClassColumn = predictions;
        } else {
            //Standard case - 2 output variables (probability distribution)
            positiveActualClassColumn = outcomes.getColumn(1);
            negativeActualClassColumn = outcomes.getColumn(0);
            positivePredictedClassColumn = predictions.getColumn(1);
        }

        //Increment global counts - actual positive/negative observed
        countActualPositive += positiveActualClassColumn.sumNumber().intValue();
        countActualNegative += negativeActualClassColumn.sumNumber().intValue();

        //Here: calculate true positive rate (TPR) vs. false positive rate (FPR) at different threshold

        for (int i = 0; i <= thresholdSteps; i++) {
            double currThreshold = i * step;

            //Work out true/false positives - do this by replacing probabilities (predictions) with 1 or 0 based on threshold
            Condition condGeq = Conditions.greaterThanOrEqual(currThreshold);
            Condition condLeq = Conditions.lessThanOrEqual(currThreshold);

            Op op = new CompareAndSet(positivePredictedClassColumn.dup(), 1.0, condGeq);
            INDArray predictedClass1 = Nd4j.getExecutioner().execAndReturn(op);
            op = new CompareAndSet(predictedClass1, 0.0, condLeq);
            predictedClass1 = Nd4j.getExecutioner().execAndReturn(op);


            //True positives: occur when positive predicted class and actual positive actual class...
            //False positive occurs when positive predicted class, but negative actual class
            INDArray isTruePositive = predictedClass1.mul(positiveActualClassColumn);       //If predicted == 1 and actual == 1 at this threshold: 1x1 = 1. 0 otherwise
            INDArray isFalsePositive = predictedClass1.mul(negativeActualClassColumn);      //If predicted == 1 and actual == 0 at this threshold: 1x1 = 1. 0 otherwise

            //Counts for this batch:
            int truePositiveCount = isTruePositive.sumNumber().intValue();
            int falsePositiveCount = isFalsePositive.sumNumber().intValue();

            //Increment counts for this thold
            CountsForThreshold thresholdCounts = counts.get(currThreshold);
            thresholdCounts.incrementTruePositive(truePositiveCount);
            thresholdCounts.incrementFalsePositive(falsePositiveCount);
        }
    }

    public List<ROCValue> getResults() {
        List<ROCValue> out = new ArrayList<>(counts.size());

        for (Map.Entry<Double, CountsForThreshold> entry : counts.entrySet()) {
            double t = entry.getKey();
            CountsForThreshold c = entry.getValue();
            double tpr = c.getCountTruePositive() / ((double) countActualPositive);
            double fpr = c.getCountFalsePositive() / ((double) countActualNegative);

            out.add(new ROCValue(t, tpr, fpr));
        }

        return out;
    }


    @AllArgsConstructor
    @Data
    public static class ROCValue {
        private final double threshold;
        private final double truePositiveRate;
        private final double falsePositiveRate;
    }

    @AllArgsConstructor
    @Data
    private static class CountsForThreshold implements Serializable {
        private double threshold;
        private long countTruePositive;
        private long countFalsePositive;

        private CountsForThreshold(double threshold) {
            this(threshold, 0, 0);
        }

        private void incrementTruePositive(long count) {
            countTruePositive += count;
        }

        private void incrementFalsePositive(long count) {
            countFalsePositive += count;
        }
    }
}
