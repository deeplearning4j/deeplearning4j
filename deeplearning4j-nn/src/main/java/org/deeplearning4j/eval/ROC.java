package org.deeplearning4j.eval;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.io.Serializable;
import java.util.*;

/**
 * ROC (Receiver Operating Characteristic) for binary classifiers, using the specified number of threshold steps.
 * <p>
 * Some ROC implementations will automatically calculate the threshold points based on the data set to give a 'smoother'
 * ROC curve (or optimal cut points for diagnostic purposes). This implementation currently uses fixed steps of size
 * 1.0 / thresholdSteps, as this allows easy implementation for batched and distributed evaluation scenarios (where the
 * full data set is not available in memory on any one machine at once).
 * <p>
 * The data is assumed to be binary classification - nColumns == 1 (single binary output variable) or nColumns == 2
 * (probability distribution over 2 classes, with column 1 being values for 'positive' examples)
 *
 * @author Alex Black
 */
@Getter
public class ROC extends BaseEvaluation<ROC> {

    private final int thresholdSteps;

    private long countActualPositive;
    private long countActualNegative;

    private final Map<Double, CountsForThreshold> counts = new LinkedHashMap<>();

    /**
     * @param thresholdSteps Number of threshold steps to use for the ROC calculation
     */
    public ROC(int thresholdSteps) {
        this.thresholdSteps = thresholdSteps;

        double step = 1.0 / thresholdSteps;
        for (int i = 0; i <= thresholdSteps; i++) {
            double currThreshold = i * step;
            counts.put(currThreshold, new CountsForThreshold(currThreshold));
        }
    }


    /**
     * Evaluate (collect statistics for) the given minibatch of data.
     * For time series (3 dimensions) use {@link #evalTimeSeries(INDArray, INDArray)} or {@link #evalTimeSeries(INDArray, INDArray, INDArray)}
     *
     * @param labels      Labels / true outcomes
     * @param predictions Predictions
     */
    public void eval(INDArray labels, INDArray predictions) {
        if (labels.rank() == 3 && predictions.rank() == 3) {
            //Assume time series input -> reshape to 2d
            evalTimeSeries(labels, predictions);
        }
        if (labels.rank() > 2 || predictions.rank() > 2 || labels.size(1) != predictions.size(1) || labels.size(1) > 2) {
            throw new IllegalArgumentException("Invalid input data shape: labels shape = " + Arrays.toString(labels.shape()) +
                    ", predictions shape = " + Arrays.toString(predictions.shape()) + "; require rank 2 array with size(1) == 1 or 2");
        }

        double step = 1.0 / thresholdSteps;
        boolean singleOutput = labels.size(1) == 1;

        INDArray positivePredictedClassColumn;
        INDArray positiveActualClassColumn;
        INDArray negativeActualClassColumn;

        if (singleOutput) {
            //Single binary variable case
            positiveActualClassColumn = labels;
            negativeActualClassColumn = labels.rsub(1.0); //1.0 - label
            positivePredictedClassColumn = predictions;
        } else {
            //Standard case - 2 output variables (probability distribution)
            positiveActualClassColumn = labels.getColumn(1);
            negativeActualClassColumn = labels.getColumn(0);
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

    /**
     * Get the ROC curve, as a set of points
     *
     * @return ROC curve, as a list of points
     */
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

    public List<PrecisionRecallPoint> getPrecisionRecallCurve(){
        //Precision: (true positive count) / (true positive count + false positive count) == true positive rate
        //Recall: (true positive count) / (true positive count + false negative count) = (TP count) / (total dataset positives)

        List<PrecisionRecallPoint> out = new ArrayList<>(counts.size());

        for (Map.Entry<Double, CountsForThreshold> entry : counts.entrySet()) {
            double t = entry.getKey();
            CountsForThreshold c = entry.getValue();
            long tpCount = c.getCountTruePositive();
            long fpCount = c.getCountFalsePositive();
            //For edge cases: http://stats.stackexchange.com/questions/1773/what-are-correct-values-for-precision-and-recall-in-edge-cases
            //precision == 1 when FP = 0 -> no incorrect positive predictions
            //recall == 1 when no dataset positives are present (got all 0 of 0 positives)
            double precision;
            if(tpCount == 0 && fpCount == 0){
                //At this threshold: no predicted positive cases
                precision = 1.0;
            } else {
                precision = tpCount / (double)(tpCount + fpCount);
            }

            double recall;
            if(countActualPositive == 0){
                recall = 1.0;
            } else {
                recall = tpCount / ((double) countActualPositive);
            }


            out.add(new PrecisionRecallPoint(c.getThreshold(), precision, recall));
        }

        return out;
    }

    /**
     * Get the ROC curve, as a set of (falsePositive, truePositive) points
     * <p>
     * Returns a 2d array of {falsePositive, truePositive values}.<br>
     * Size is [2][thresholdSteps], with out[0][.] being false positives, and out[1][.] being true positives
     *
     * @return ROC curve as double[][]
     */
    public double[][] getResultsAsArray() {
        double[][] out = new double[2][thresholdSteps+1];
        int i = 0;
        for (Map.Entry<Double, CountsForThreshold> entry : counts.entrySet()) {
            CountsForThreshold c = entry.getValue();
            double tpr = c.getCountTruePositive() / ((double) countActualPositive);
            double fpr = c.getCountFalsePositive() / ((double) countActualNegative);

            out[0][i] = fpr;
            out[1][i] = tpr;
            i++;
        }
        return out;
    }

    /**
     * Calculate the AUC - Area Under Curve<br>
     * Utilizes trapezoidal integration internally
     *
     * @return AUC
     */
    public double calculateAUC() {
        //Calculate AUC using trapezoidal rule
        List<ROCValue> list = getResults();

        //Given the points
        double auc = 0.0;
        for (int i = 0; i < list.size() - 1; i++) {
            ROCValue left = list.get(i);
            ROCValue right = list.get(i + 1);

            //y axis: TPR
            //x axis: FPR
            double deltaX = Math.abs(right.getFalsePositiveRate() - left.getFalsePositiveRate());   //Iterating in threshold order, so FPR decreases as threshold increases
            double avg = (left.getTruePositiveRate() + right.getTruePositiveRate()) / 2.0;

            auc += deltaX * avg;
        }
        return auc;
    }

    /**
     * Merge this ROC instance with another.
     * This ROC instance is modified, by adding the stats from the other instance.
     *
     * @param other ROC instance to combine with this one
     */
    @Override
    public void merge(ROC other){
        if(this.thresholdSteps != other.thresholdSteps){
            throw new UnsupportedOperationException("Cannot merge ROC instances with different numbers of threshold steps ("
                    + this.thresholdSteps + " vs. " + other.thresholdSteps + ")");
        }
        this.countActualPositive += other.countActualPositive;
        this.countActualNegative += other.countActualNegative;
        for(Double d : this.counts.keySet()){
            CountsForThreshold cft = this.counts.get(d);
            CountsForThreshold otherCft = other.counts.get(d);
            cft.countTruePositive += otherCft.countTruePositive;
            cft.countFalsePositive += otherCft.countFalsePositive;
        }
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
    public static class PrecisionRecallPoint {
        private final double classiferThreshold;
        private final double precision;
        private final double recall;
    }

    @AllArgsConstructor
    @Data
    public static class CountsForThreshold implements Serializable, Cloneable {
        private double threshold;
        private long countTruePositive;
        private long countFalsePositive;

        public CountsForThreshold(double threshold) {
            this(threshold, 0, 0);
        }

        public void incrementTruePositive(long count) {
            countTruePositive += count;
        }

        public void incrementFalsePositive(long count) {
            countFalsePositive += count;
        }

        @Override
        public CountsForThreshold clone(){
            return new CountsForThreshold(threshold, countTruePositive, countFalsePositive);
        }
    }
}
