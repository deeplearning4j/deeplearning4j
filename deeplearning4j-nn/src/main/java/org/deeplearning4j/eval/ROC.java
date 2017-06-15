package org.deeplearning4j.eval;

import lombok.*;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.shade.jackson.annotation.JsonIgnore;

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
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor
public class ROC extends BaseEvaluation<ROC> {

    private static final int EXACT_ALLOC_BLOCK_SIZE = 10000;

    private  int thresholdSteps;
    private long countActualPositive;
    private long countActualNegative;
    private final Map<Double, CountsForThreshold> counts = new LinkedHashMap<>();

    private Double auc;
    private Double auprc;

    private boolean isExact;
    private INDArray probAndLabel;
    private int exampleCount = 0;

    /**
     * @param thresholdSteps Number of threshold steps to use for the ROC calculation. If set to 0: use exact calculation
     */
    public ROC(int thresholdSteps) {

        if(thresholdSteps > 0){
            this.thresholdSteps = thresholdSteps;

            double step = 1.0 / thresholdSteps;
            for (int i = 0; i <= thresholdSteps; i++) {
                double currThreshold = i * step;
                counts.put(currThreshold, new CountsForThreshold(currThreshold));
            }

            isExact = false;
        } else {
            //Exact

            isExact = true;
        }
    }

    @Override
    public void reset() {
        countActualPositive = 0L;
        countActualNegative = 0L;
        counts.clear();

        if(isExact){
            probAndLabel = null;
            exampleCount = 0;
        } else {
            double step = 1.0 / thresholdSteps;
            for (int i = 0; i <= thresholdSteps; i++) {
                double currThreshold = i * step;
                counts.put(currThreshold, new CountsForThreshold(currThreshold));
            }
        }

        auc = null;
        auprc = null;
    }

    @Override
    public String stats() {
        return "AUC: [" + calculateAUC() + "]";
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
        if (labels.rank() > 2 || predictions.rank() > 2 || labels.size(1) != predictions.size(1)
                        || labels.size(1) > 2) {
            throw new IllegalArgumentException("Invalid input data shape: labels shape = "
                            + Arrays.toString(labels.shape()) + ", predictions shape = "
                            + Arrays.toString(predictions.shape()) + "; require rank 2 array with size(1) == 1 or 2");
        }

        double step = 1.0 / thresholdSteps;
        boolean singleOutput = labels.size(1) == 1;

        if(isExact){
            //Exact approach: simply add them to the storage for later computation/use

            if(probAndLabel == null){
                //Do initial allocation
                int initialSize = Math.max(labels.size(0), EXACT_ALLOC_BLOCK_SIZE);
                probAndLabel = Nd4j.create(initialSize, 2); //First col: probability of class 1. Second col: "is class 1"
            }

            //Allocate a larger array if necessary
            if(exampleCount + labels.size(0) >= probAndLabel.size(0)){
                int newSize = probAndLabel.size(0) + Math.max(EXACT_ALLOC_BLOCK_SIZE, labels.size(0));
                INDArray newProbAndLabel = Nd4j.create(newSize, 2);
                newProbAndLabel.assign(probAndLabel.get(NDArrayIndex.interval(0,exampleCount), NDArrayIndex.all()));
                probAndLabel = newProbAndLabel;
            }

            //put values
            INDArray probClass1;
            INDArray labelClass1;
            if(singleOutput){
                probClass1 = predictions;
                labelClass1 = labels;
            } else {
                probClass1 = predictions.getColumn(1);
                labelClass1 = labels.getColumn(1);
            }
            int currMinibatchSize = labels.size(0);
            probAndLabel.get(NDArrayIndex.interval(exampleCount, exampleCount+currMinibatchSize), NDArrayIndex.point(0))
                    .assign(probClass1);

            probAndLabel.get(NDArrayIndex.interval(exampleCount, exampleCount+currMinibatchSize), NDArrayIndex.point(0))
                    .assign(labelClass1);

            int countClass1CurrMinibatch = labelClass1.sumNumber().intValue();
            countActualPositive += countClass1CurrMinibatch;
            countActualNegative += labels.size(0) - countClass1CurrMinibatch;
            exampleCount += labels.size(0);
        } else {
            //Thresholded approach
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

            INDArray ppc = null;
            INDArray itp = null;
            INDArray ifp = null;
            for (int i = 0; i <= thresholdSteps; i++) {
                double currThreshold = i * step;

                //Work out true/false positives - do this by replacing probabilities (predictions) with 1 or 0 based on threshold
                Condition condGeq = Conditions.greaterThanOrEqual(currThreshold);
                Condition condLeq = Conditions.lessThanOrEqual(currThreshold);

                if (ppc == null) {
                    ppc = positivePredictedClassColumn.dup(positiveActualClassColumn.ordering());
                } else {
                    ppc.assign(positivePredictedClassColumn);
                }
                Op op = new CompareAndSet(ppc, 1.0, condGeq);
                INDArray predictedClass1 = Nd4j.getExecutioner().execAndReturn(op);
                op = new CompareAndSet(predictedClass1, 0.0, condLeq);
                predictedClass1 = Nd4j.getExecutioner().execAndReturn(op);


                //True positives: occur when positive predicted class and actual positive actual class...
                //False positive occurs when positive predicted class, but negative actual class
                INDArray isTruePositive;    // = predictedClass1.mul(positiveActualClassColumn); //If predicted == 1 and actual == 1 at this threshold: 1x1 = 1. 0 otherwise
                INDArray isFalsePositive;   // = predictedClass1.mul(negativeActualClassColumn); //If predicted == 1 and actual == 0 at this threshold: 1x1 = 1. 0 otherwise
                if (i == 0) {
                    isTruePositive = predictedClass1.mul(positiveActualClassColumn);
                    isFalsePositive = predictedClass1.mul(negativeActualClassColumn);
                    itp = isTruePositive;
                    ifp = isFalsePositive;
                } else {
                    isTruePositive = Nd4j.getExecutioner().execAndReturn(new MulOp(predictedClass1, positiveActualClassColumn, itp));
                    isFalsePositive = Nd4j.getExecutioner().execAndReturn(new MulOp(predictedClass1, negativeActualClassColumn, ifp));
                }

                //Counts for this batch:
                int truePositiveCount = isTruePositive.sumNumber().intValue();
                int falsePositiveCount = isFalsePositive.sumNumber().intValue();

                //Increment counts for this thold
                CountsForThreshold thresholdCounts = counts.get(currThreshold);
                thresholdCounts.incrementTruePositive(truePositiveCount);
                thresholdCounts.incrementFalsePositive(falsePositiveCount);
            }
        }

        auc = null;
        auprc = null;
    }

    /**
     * Get the ROC curve, as a set of points
     *
     * @return ROC curve, as a list of points
     */
    @JsonIgnore
    public List<ROCValue> getResults() {
        if(isExact){
            //Might change this in the future
            throw new IllegalStateException("Cannot get points from exact ROC calculation");
        }

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

    @JsonIgnore
    public List<PrecisionRecallPoint> getPrecisionRecallCurve() {
        if(isExact){
            //Might change this in the future
            throw new IllegalStateException("Cannot get points from exact ROC calculation");
        }

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
            if (tpCount == 0 && fpCount == 0) {
                //At this threshold: no predicted positive cases
                precision = 1.0;
            } else {
                precision = tpCount / (double) (tpCount + fpCount);
            }

            double recall;
            if (countActualPositive == 0) {
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
    @JsonIgnore
    public double[][] getResultsAsArray() {
        if(isExact){
            //Might change this in the future
            throw new IllegalStateException("Cannot get points from exact ROC calculation");
        }

        double[][] out = new double[2][thresholdSteps + 1];
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
     * Calculate the AUROC - Area Under ROC Curve<br>
     * Utilizes trapezoidal integration internally
     *
     * @return AUC
     */
    public double calculateAUC() {
        if(auc != null){
            return auc;
        }


        if(isExact){
            //http://www.cs.waikato.ac.nz/~remco/roc.pdf section 2

            INDArray sorted = Nd4j.sortRows(probAndLabel, 0, true);
            INDArray isPositive = sorted.getColumn(0);
            INDArray isNegative = sorted.getColumn(0).rsub(1.0);

            INDArray cumSumPos = isPositive.cumsum(-1);
            INDArray cumSumNeg = isNegative.cumsum(-1);

            int totalPositives = isPositive.sumNumber().intValue();
            int totalNegatives = isPositive.length() - totalPositives;

            //Cumulative sum:
            int length = isNegative.length();

            //Here: have "length" points, but we want these indexed as 1 to n
            double aucSum = 0.0;
            for( int i=0; i<=length; i++ ){
                //Y axis: TPR = sum(TP at current threshold) / totalPositives
                //X axis: FPR = sum(FP at current threshold) / totalNegatives

                double x_fpr;
                double x_fpr_next;

                double y_tpr;
                double y_tpr_next;

                if( i > 0 ){
                    x_fpr = cumSumNeg.getDouble(i-1) / totalNegatives;
                    y_tpr = cumSumPos.getDouble(i-1) / totalPositives;
                } else {
                    y_tpr = 0.0;
                    x_fpr = 0.0;
                }

                if( i < length){
                    x_fpr_next = cumSumNeg.getDouble(i) / totalNegatives;
                    y_tpr_next = cumSumPos.getDouble(i) / totalPositives;
                } else {
                    x_fpr_next = 1.0;
                    y_tpr_next = 1.0;
                }


                double dx = (x_fpr_next - x_fpr);
                if(dx > 0.0) {
                    aucSum += (y_tpr_next + y_tpr) / (2.0 * dx);
                }
            }

            this.auc = aucSum;
            return aucSum;
        } else {
            //Calculate AUC using trapezoidal rule
            List<ROCValue> list = getResults();

            //Given the points
            double auc = 0.0;
            for (int i = 0; i < list.size() - 1; i++) {
                ROCValue left = list.get(i);
                ROCValue right = list.get(i + 1);

                //y axis: TPR
                //x axis: FPR
                double deltaX = Math.abs(right.getFalsePositiveRate() - left.getFalsePositiveRate()); //Iterating in threshold order, so FPR decreases as threshold increases
                double avg = (left.getTruePositiveRate() + right.getTruePositiveRate()) / 2.0;

                auc += deltaX * avg;
            }

            this.auc = auc;
            return auc;
        }
    }

    /**
     * Calculate the area under the precision/recall curve - aka AUCPR
     *
     * @return
     */
    public double calculateAUCPR(){

        if(auprc != null){
            return auprc;
        }

        if(isExact){
            throw new UnsupportedOperationException("Not yet implemented");
        }

        List<PrecisionRecallPoint> prCurve = getPrecisionRecallCurve();
        //Sorting by recall is unnecessary: recall increases as threshold increases, and PR curve points are
        //sorted by threshold by default
        //X axis: recall
        //Y axis: precision

        //Trapezoidal integration
        double aucpr = 0.0;
        for (int i = 0; i < prCurve.size()-1; i++) {
            double x0 = prCurve.get(i).getRecall();
            double x1 = prCurve.get(i+1).getRecall();
            double deltaX = Math.abs(x1 - x0);  //Going from highest recall (at 0 threshold) to lowest recall (at 1.0 threshold)
            double y0 = prCurve.get(i).getPrecision();
            double y1 = prCurve.get(i+1).getPrecision();
            double avgY = (y0+y1) / 2.0;

            aucpr += deltaX*avgY;
        }

        return aucpr;
    }

    /**
     * Merge this ROC instance with another.
     * This ROC instance is modified, by adding the stats from the other instance.
     *
     * @param other ROC instance to combine with this one
     */
    @Override
    public void merge(ROC other) {
        if (this.thresholdSteps != other.thresholdSteps) {
            throw new UnsupportedOperationException(
                            "Cannot merge ROC instances with different numbers of threshold steps ("
                                            + this.thresholdSteps + " vs. " + other.thresholdSteps + ")");
        }
        this.countActualPositive += other.countActualPositive;
        this.countActualNegative += other.countActualNegative;
        this.auc = null;
        this.auprc = null;

        if(isExact){

            if(other.exampleCount == 0){
                return;
            }

            if(this.exampleCount == 0){
                this.exampleCount = other.exampleCount;
                this.probAndLabel = other.probAndLabel;
                return;
            }

            if(this.exampleCount + other.exampleCount > this.probAndLabel.size(0)){
                //Allocate new array
                int newSize = this.probAndLabel.size(0) + Math.max(other.probAndLabel.size(0), EXACT_ALLOC_BLOCK_SIZE);
                INDArray newProbAndLabel = Nd4j.create(newSize, 2);
                newProbAndLabel.assign(probAndLabel.get(NDArrayIndex.interval(0,exampleCount), NDArrayIndex.all()));
                probAndLabel = newProbAndLabel;
            }

            INDArray toPut = other.probAndLabel.get(NDArrayIndex.interval(0, other.exampleCount), NDArrayIndex.all());
            probAndLabel.put(new INDArrayIndex[]{NDArrayIndex.interval(exampleCount, exampleCount + other.exampleCount),
                    NDArrayIndex.all()}, toPut);

            this.exampleCount += other.exampleCount;
        } else {
            for (Double d : this.counts.keySet()) {
                CountsForThreshold cft = this.counts.get(d);
                CountsForThreshold otherCft = other.counts.get(d);
                cft.countTruePositive += otherCft.countTruePositive;
                cft.countFalsePositive += otherCft.countFalsePositive;
            }
        }
    }


    @AllArgsConstructor
    @Data
    @NoArgsConstructor
    public static class ROCValue {
        private  double threshold;
        private  double truePositiveRate;
        private  double falsePositiveRate;
    }

    @AllArgsConstructor
    @Data
    @NoArgsConstructor
    public static class PrecisionRecallPoint {
        private  double classiferThreshold;
        private  double precision;
        private  double recall;
    }

    @AllArgsConstructor
    @Data
    @NoArgsConstructor
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
        public CountsForThreshold clone() {
            return new CountsForThreshold(threshold, countTruePositive, countFalsePositive);
        }
    }
}
