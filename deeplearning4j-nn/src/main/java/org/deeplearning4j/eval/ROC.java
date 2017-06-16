package org.deeplearning4j.eval;

import lombok.*;
import org.apache.commons.lang3.ArrayUtils;
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
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor
@Data
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
    private boolean rocRemoveRedundantPts;

    /**
     * @param thresholdSteps Number of threshold steps to use for the ROC calculation. If set to 0: use exact calculation
     */
    public ROC(int thresholdSteps) {
        this(thresholdSteps, true);
    }


    public ROC(int thresholdSteps, boolean rocRemoveRedundantPts) {

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
        this.rocRemoveRedundantPts = rocRemoveRedundantPts;
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
     * @deprecated Use {@link #getPrecisionRecallCurveAsArray()}
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
        int n = asArray[0].length;
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
    @JsonIgnore
    public double[][] getPrecisionRecallCurveAsArray() {

        double[] thresholdOut;
        double[] precisionOut;
        double[] recallOut;

        if (isExact) {
            INDArray pl = getProbAndLabelUsed();
            INDArray sorted = Nd4j.sortRows(pl, 0, false);
            INDArray isPositive = sorted.getColumn(1);
            INDArray isNegative = isPositive.rsub(1.0);

            INDArray cumSumPos = isPositive.cumsum(-1);
            INDArray cumSumNeg = isNegative.cumsum(-1);
            int length = sorted.size(0);

            /*
            Sort descending. As we iterate: decrease probability threshold T... all values <= T are predicted
            as class 0, all others are predicted as class 1

            Precision:  sum(TP) / sum(predicted pos at threshold)
            Recall:     sum(TP) / total actual positives

            predicted positive at threshold: # values <= threshold, i.e., just i
             */

            INDArray t = Nd4j.create(new int[]{length+2,1});
            t.put(new INDArrayIndex[]{NDArrayIndex.interval(1,length+1),NDArrayIndex.all()}, sorted.getColumn(0));

            INDArray linspace = Nd4j.linspace(1, length, length);
            INDArray precision = cumSumPos.div(linspace);
            INDArray prec = Nd4j.create(new int[]{length+2,1});
            prec.put(new INDArrayIndex[]{NDArrayIndex.interval(1,length+1),NDArrayIndex.all()}, precision);

            //Recall/TPR
            INDArray rec = Nd4j.create(new int[]{length+2,1});
            rec.put(new INDArrayIndex[]{NDArrayIndex.interval(1,length+1),NDArrayIndex.all()}, cumSumPos.div(countActualPositive));

            //Edge cases
            t.putScalar(0, 0, 1.0);
            prec.putScalar(0, 0, 1.0);
            rec.putScalar(0, 0, 0.0);
            prec.putScalar(length+1, 0, cumSumPos.getDouble(cumSumPos.length()-1) / length);
            rec.putScalar(length+1, 0, 1.0);

            thresholdOut = t.data().asDouble();
            precisionOut = prec.data().asDouble();
            recallOut = rec.data().asDouble();

            //Finally: 2 things to do
            //(a) Reverse order: lowest to highest threshold
            //(b) remove unnecessary/rendundant points (doesn't affect graph or AUPRC)

            ArrayUtils.reverse(thresholdOut);
            ArrayUtils.reverse(precisionOut);
            ArrayUtils.reverse(recallOut);

            if(rocRemoveRedundantPts) {
                double[][] temp = removeRedundant(thresholdOut, precisionOut, recallOut);
                thresholdOut = temp[0];
                precisionOut = temp[1];
                recallOut = temp[2];
            }
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
            //Sort ascending. As we decrease threshold, more are predicted positive.
            //if(prob <= threshold> predict 0, otherwise predict 1
            //So, as we iterate from i=0..length, first 0 to i (inclusive) are predicted class 1, all others are predicted class 0
            INDArray pl = getProbAndLabelUsed();
            INDArray sorted = Nd4j.sortRows(pl, 0, false);
            INDArray isPositive = sorted.getColumn(1);
            INDArray isNegative = sorted.getColumn(1).rsub(1.0);

            INDArray cumSumPos = isPositive.cumsum(-1);
            INDArray cumSumNeg = isNegative.cumsum(-1);
            int length = sorted.size(0);

            INDArray t = Nd4j.create(new int[]{length+2,1});
            t.put(new INDArrayIndex[]{NDArrayIndex.interval(1,length+1),NDArrayIndex.all()}, sorted.getColumn(0));

            INDArray fpr = Nd4j.create(new int[]{length+2,1});
            fpr.put(new INDArrayIndex[]{NDArrayIndex.interval(1,length+1),NDArrayIndex.all()}, cumSumNeg.div(countActualNegative));

            INDArray tpr = Nd4j.create(new int[]{length+2,1});
            tpr.put(new INDArrayIndex[]{NDArrayIndex.interval(1,length+1),NDArrayIndex.all()}, cumSumPos.div(countActualPositive));

            //Edge cases
            t.putScalar(0, 0, 1.0);
            fpr.putScalar(0, 0, 0.0);
            tpr.putScalar(0, 0, 0.0);
            fpr.putScalar(length+1, 0, 1.0);
            tpr.putScalar(length+1, 0, 1.0);


            double[] x_fpr_out = fpr.data().asDouble();
            double[] y_tpr_out = tpr.data().asDouble();
            double[] tOut = t.data().asDouble();

            //Note: we can have multiple FPR for a given TPR, and multiple TPR for a given FPR
            //These can be omitted, without changing the area (as long as we keep the edge points)
            if(rocRemoveRedundantPts) {
//                double[] t_compacted = new double[tOut.length];
//                double[] x_fpr_compacted = new double[x_fpr_out.length];
//                double[] y_tpr_compacted = new double[y_tpr_out.length];
//                int lastOutPos = -1;
//                for (int i = 0; i < tOut.length; i++) {
//
//                    boolean keep;
//                    if (i == 0 || i == tOut.length - 1) {
//                        keep = true;
//                    } else {
//                        boolean ommitSameTPR = y_tpr_out[i - 1] == y_tpr_out[i] && y_tpr_out[i] == y_tpr_out[i + 1];
//                        boolean ommitSameFPR = x_fpr_out[i - 1] == x_fpr_out[i] && x_fpr_out[i] == x_fpr_out[i + 1];
//                        keep = !ommitSameFPR && !ommitSameTPR;
//                    }
//
//                    if (keep) {
//                        lastOutPos++;
//                        t_compacted[lastOutPos] = tOut[i];
//                        y_tpr_compacted[lastOutPos] = y_tpr_out[i];
//                        x_fpr_compacted[lastOutPos] = x_fpr_out[i];
//                    }
//                }
//
//                if (lastOutPos < x_fpr_out.length - 1) {
//                    tOut = Arrays.copyOfRange(t_compacted, 0, lastOutPos + 1);
//                    x_fpr_out = Arrays.copyOfRange(x_fpr_compacted, 0, lastOutPos + 1);
//                    y_tpr_out = Arrays.copyOfRange(y_tpr_compacted, 0, lastOutPos + 1);
//                }

                double[][] temp = removeRedundant(tOut, x_fpr_out, y_tpr_out);
                tOut = temp[0];
                x_fpr_out = temp[1];
                y_tpr_out = temp[2];
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

    private static double[][] removeRedundant(double[] threshold, double[] x, double[] y){
        double[] t_compacted = new double[threshold.length];
        double[] x_compacted = new double[x.length];
        double[] y_compacted = new double[y.length];
        int lastOutPos = -1;
        for (int i = 0; i < threshold.length; i++) {

            boolean keep;
            if (i == 0 || i == threshold.length - 1) {
                keep = true;
            } else {
                boolean ommitSameY = y[i - 1] == y[i] && y[i] == y[i + 1];
                boolean ommitSameX = x[i - 1] == x[i] && x[i] == x[i + 1];
                keep = !ommitSameX && !ommitSameY;
            }

            if (keep) {
                lastOutPos++;
                t_compacted[lastOutPos] = threshold[i];
                y_compacted[lastOutPos] = y[i];
                x_compacted[lastOutPos] = x[i];
            }
        }

        if (lastOutPos < x.length - 1) {
            t_compacted = Arrays.copyOfRange(t_compacted, 0, lastOutPos + 1);
            x_compacted = Arrays.copyOfRange(x_compacted, 0, lastOutPos + 1);
            y_compacted = Arrays.copyOfRange(y_compacted, 0, lastOutPos + 1);
        }

        return new double[][]{t_compacted, x_compacted, y_compacted};
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
            double fprLeft = rocAsArray[1][i];
            double tprLeft = rocAsArray[2][i];
            double fprRight = rocAsArray[1][i + 1];
            double tprRight = rocAsArray[2][i + 1];

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
