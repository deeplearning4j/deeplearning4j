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
import org.nd4j.linalg.string.NDArrayStrings;
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

    private int thresholdSteps;
    private long countActualPositive;
    private long countActualNegative;
    private final Map<Double, CountsForThreshold> counts = new LinkedHashMap<>();

    @Getter(AccessLevel.PRIVATE)
    private Double auc;
    @Getter(AccessLevel.PRIVATE)
    private Double auprc;

    private boolean isExact;
    private INDArray probAndLabel;
    private int exampleCount = 0;

    /**
     * @param thresholdSteps Number of threshold steps to use for the ROC calculation. If set to 0: use exact calculation
     */
    public ROC(int thresholdSteps) {

        if (thresholdSteps > 0) {
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

    protected INDArray getProbAndLabelUsed() {
        if (probAndLabel == null || exampleCount == 0) {
            return null;
        }
        return probAndLabel.get(NDArrayIndex.interval(0, exampleCount), NDArrayIndex.all());
    }

    @Override
    public void reset() {
        countActualPositive = 0L;
        countActualNegative = 0L;
        counts.clear();

        if (isExact) {
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

        if (isExact) {
            //Exact approach: simply add them to the storage for later computation/use

            if (probAndLabel == null) {
                //Do initial allocation
                int initialSize = Math.max(labels.size(0), EXACT_ALLOC_BLOCK_SIZE);
                probAndLabel = Nd4j.create(new int[]{initialSize, 2}, 'c'); //First col: probability of class 1. Second col: "is class 1"
            }

            //Allocate a larger array if necessary
            if (exampleCount + labels.size(0) >= probAndLabel.size(0)) {
                int newSize = probAndLabel.size(0) + Math.max(EXACT_ALLOC_BLOCK_SIZE, labels.size(0));
                INDArray newProbAndLabel = Nd4j.create(new int[]{newSize, 2}, 'c');
                newProbAndLabel.assign(probAndLabel.get(NDArrayIndex.interval(0, exampleCount), NDArrayIndex.all()));
                probAndLabel = newProbAndLabel;
            }

            //put values
            INDArray probClass1;
            INDArray labelClass1;
            if (singleOutput) {
                probClass1 = predictions;
                labelClass1 = labels;
            } else {
                probClass1 = predictions.getColumn(1);
                labelClass1 = labels.getColumn(1);
            }
            int currMinibatchSize = labels.size(0);
            probAndLabel.get(NDArrayIndex.interval(exampleCount, exampleCount + currMinibatchSize), NDArrayIndex.point(0))
                    .assign(probClass1);

            probAndLabel.get(NDArrayIndex.interval(exampleCount, exampleCount + currMinibatchSize), NDArrayIndex.point(1))
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
     * @deprecated Use {@link #getRocCurve()}
     */
    @JsonIgnore
    @Deprecated
    public List<ROCValue> getResults() {
        return getRocCurve();
    }

    /**
     * Get the ROC curve, as a set of points
     *
     * @return ROC curve, as a list of points
     * @deprecated Use {@link #getPrecisionRecallCurveAsArray()}
     */
    @JsonIgnore
    @Deprecated
    public List<ROCValue> getRocCurve() {
        List<ROCValue> out = new ArrayList<>();

        double[][] asArray = getRocCurveAsArray();  //Threshold, fpr, tpr
        int n = asArray.length;
        for (int i = 0; i < n; i++) {
            out.add(new ROCValue(asArray[0][i], asArray[2][i], asArray[1][i])); //ROCValue: thresh, tpr, fpr
        }
        return out;
    }

    /**
     * @deprecated Use {@link #getPrecisionRecallCurveAsArray()}
     */
    @JsonIgnore
    @Deprecated
    public List<PrecisionRecallPoint> getPrecisionRecallCurve() {
        double[][] asArr = getPrecisionRecallCurveAsArray();
        int length = asArr[0].length;
        List<PrecisionRecallPoint> out = new ArrayList<>(length);
        for (int i = 0; i < length; i++) {
            out.add(new PrecisionRecallPoint(asArr[0][i], asArr[1][i], asArr[2][i]));
        }
        return out;
    }

    /**
     * Get the precision recall curve as array.
     * return[0] = threshold array<br>
     * return[1] = precision array<br>
     * return[2] = recall array<br>
     *
     * @return
     */
    public double[][] getPrecisionRecallCurveAsArray() {

        double[] thresholdOut;
        double[] precisionOut;
        double[] recallOut;

        if (isExact) {
            INDArray pl = getProbAndLabelUsed();
            INDArray sorted = Nd4j.sortRows(pl, 0, true);
            INDArray isPositive = sorted.getColumn(0);

            INDArray cumSumPos = isPositive.cumsum(-1);
            int numPredictions = isPositive.length();

            //Num predictions + 2: (+2) is for 0 and 1 threshold
            thresholdOut = new double[numPredictions + 2];
            precisionOut = new double[numPredictions + 2];
            recallOut = new double[numPredictions + 2];

            //Precision: sum(TP) / sum(predicted pos at threshold)
            //Recall: sum(TP) / total actual positives

            //Edge case: threshold of 0.0. All predicted negative, recall = 0.0
            // TP = 0, PP at threshold = 0 -> 0 / 0 -> use 1.0
            //But Recall = 0
            precisionOut[0] = 1.0;

            for (int i = 1; i <= numPredictions; i++) { //Start at 1 -> threshold/precision/recall[0] == 0.0 by design
                double tpCountAtThreshold = cumSumPos.getDouble(i - 1);
                thresholdOut[i] = sorted.getDouble(i - 1, 0);
                precisionOut[i] = tpCountAtThreshold / i;
                recallOut[i] = tpCountAtThreshold / countActualPositive;
            }

            //Threshold of 1.0: all predicted as positive
            thresholdOut[numPredictions + 1] = 1.0;
            precisionOut[numPredictions + 1] = countActualPositive / (double) numPredictions;
            recallOut[numPredictions + 1] = 1.0;
        } else {
            thresholdOut = new double[counts.size()];
            precisionOut = new double[counts.size()];
            recallOut = new double[counts.size()];

            int i = 0;
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

                thresholdOut[i] = c.getThreshold();
                precisionOut[i] = precision;
                recallOut[i] = recall;
                i++;
            }

        }
        return new double[][]{thresholdOut, precisionOut, recallOut};
    }

    /**
     * @deprecated Use {@link #getRocCurveAsArray()}
     */
    @JsonIgnore
    @Deprecated
    public double[][] getResultsAsArray() {
        return getRocCurveAsArray();
    }

    /**
     * Get the ROC curve, as a set of (threshold, falsePositive, truePositive) points
     * <p>
     * Returns a 2d array of {threshold, falsePositive, truePositive values}.<br>
     * Size is [3][thresholdSteps], with out[0][.] being threshold, out[1][.] being false positives,
     * and out[2][.] being true positives
     *
     * @return ROC curve as double[][]
     */
    @JsonIgnore
    public double[][] getRocCurveAsArray() {

        if (isExact) {
            INDArray pl = getProbAndLabelUsed();
            INDArray sorted = Nd4j.sortRows(pl, 0, true);
            INDArray isPositive = sorted.getColumn(1);
            INDArray isNegative = sorted.getColumn(1).rsub(1.0);

//            System.out.println(sorted);
            System.out.println(new NDArrayStrings(8).format(sorted));

            INDArray cumSumPos = isPositive.cumsum(-1);
            INDArray cumSumNeg = isNegative.cumsum(-1);

            int totalPositives = isPositive.sumNumber().intValue();
            int totalNegatives = isPositive.length() - totalPositives;
            int length = isNegative.length();

            double[] tOut = new double[length + 2];
            double[] x_fpr_out = new double[length + 2];
            double[] y_tpr_out = new double[length + 2];
            int lastOutPos = 0;
            for (int i = 1; i <= length; i++) { //Start at 1 -> xOut[0] and yOut[0] == 0.0 by design
                //Y axis: TPR = sum(TP at current threshold) / totalPositives
                //X axis: FPR = sum(FP at current threshold) / totalNegatives

                tOut[i] = pl.getDouble(i - 1, 0);
                double x_fpr = cumSumNeg.getDouble(i - 1) / totalNegatives;
                double y_tpr = cumSumPos.getDouble(i - 1) / totalPositives;

                lastOutPos++;
                x_fpr_out[lastOutPos] = x_fpr;
                y_tpr_out[lastOutPos] = y_tpr;
            }

            lastOutPos++;
            x_fpr_out[lastOutPos] = 1.0;
            y_tpr_out[lastOutPos] = 1.0;


            //Note: we can have multiple FPR for a given TPR, and multiple TPR for a given FPR
            //These can be omitted, without changing the area (as long as we keep the edge points)
            double[] t_compacted = new double[tOut.length];
            double[] x_fpr_compacted = new double[x_fpr_out.length];
            double[] y_tpr_compacted = new double[y_tpr_out.length];
            lastOutPos = -1;
            for (int i = 0; i < tOut.length; i++) {

                boolean keep;
                if(i == 0 || i == tOut.length -1){
                    keep = true;
                } else {
                    boolean ommitSameTPR = y_tpr_out[i - 1] == y_tpr_out[i] && y_tpr_out[i] == y_tpr_out[i + 1];
                    boolean ommitSameFPR = x_fpr_out[i - 1] == x_fpr_out[i] && x_fpr_out[i] == x_fpr_out[i + 1];
                    keep = !ommitSameFPR && !ommitSameTPR;
                }

                if (keep) {
                    lastOutPos++;
                    t_compacted[lastOutPos] = tOut[i];
                    y_tpr_compacted[lastOutPos] = y_tpr_out[i];
                    x_fpr_compacted[lastOutPos] = x_fpr_out[i];
                }
            }


            if (lastOutPos < x_fpr_out.length-1) {
                tOut = Arrays.copyOfRange(t_compacted, 0, lastOutPos+1);
                x_fpr_out = Arrays.copyOfRange(x_fpr_compacted, 0, lastOutPos+1);
                y_tpr_out = Arrays.copyOfRange(y_tpr_compacted, 0, lastOutPos+1);
            }

            return new double[][]{tOut, x_fpr_out, y_tpr_out};
        } else {

            double[][] out = new double[3][thresholdSteps + 1];
            int i = 0;
            for (Map.Entry<Double, CountsForThreshold> entry : counts.entrySet()) {
                CountsForThreshold c = entry.getValue();
                double tpr = c.getCountTruePositive() / ((double) countActualPositive);
                double fpr = c.getCountFalsePositive() / ((double) countActualNegative);

                out[0][i] = c.getThreshold();
                out[1][i] = fpr;
                out[2][i] = tpr;
                i++;
            }
            return out;
        }
    }

    /**
     * Calculate the AUROC - Area Under ROC Curve<br>
     * Utilizes trapezoidal integration internally
     *
     * @return AUC
     */
    public double calculateAUC() {
        if (auc != null) {
            return auc;
        }

        //Calculate AUC using trapezoidal rule
        double[][] rocAsArray = getRocCurveAsArray();
        int nPoints = rocAsArray[0].length;

        //Given the points
        double auc = 0.0;
        for (int i = 0; i < nPoints - 1; i++) {
            double fprLeft = rocAsArray[0][i];
            double tprLeft = rocAsArray[1][i];
            double fprRight = rocAsArray[0][i + 1];
            double tprRight = rocAsArray[1][i + 1];

            //y axis: TPR
            //x axis: FPR
            double deltaX = Math.abs(fprRight - fprLeft); //Iterating in threshold order, so FPR decreases as threshold increases
            double avg = (tprRight + tprLeft) / 2.0;

            auc += deltaX * avg;
        }

        this.auc = auc;
        return auc;
    }

    /**
     * Calculate the area under the precision/recall curve - aka AUCPR
     *
     * @return
     */
    public double calculateAUCPR() {

        if (auprc != null) {
            return auprc;
        }

        double[][] prcurve = getPrecisionRecallCurveAsArray();
        int n = prcurve[0].length;

        double prArea = 0.0;
        for (int i = 0; i < n - 1; i++) {
            double pLeft = prcurve[1][i];
            double rLeft = prcurve[2][i];
            double pRight = prcurve[1][i + 1];
            double rRight = prcurve[2][i + 1];

            double deltaX = Math.abs(rLeft - rRight);  //Going from highest recall (at 0 threshold) to lowest recall (at 1.0 threshold)
            if (deltaX == 0) {
                continue;
            }
            double avgY = (pLeft + pRight) / 2.0;
            prArea += deltaX * avgY;
        }

        auprc = prArea;
        return prArea;
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

        if (isExact) {

            if (other.exampleCount == 0) {
                return;
            }

            if (this.exampleCount == 0) {
                this.exampleCount = other.exampleCount;
                this.probAndLabel = other.probAndLabel;
                return;
            }

            if (this.exampleCount + other.exampleCount > this.probAndLabel.size(0)) {
                //Allocate new array
                int newSize = this.probAndLabel.size(0) + Math.max(other.probAndLabel.size(0), EXACT_ALLOC_BLOCK_SIZE);
                INDArray newProbAndLabel = Nd4j.create(newSize, 2);
                newProbAndLabel.assign(probAndLabel.get(NDArrayIndex.interval(0, exampleCount), NDArrayIndex.all()));
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
        private double threshold;
        private double truePositiveRate;
        private double falsePositiveRate;
    }

    @AllArgsConstructor
    @Data
    @NoArgsConstructor
    public static class PrecisionRecallPoint {
        private double classiferThreshold;
        private double precision;
        private double recall;
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
