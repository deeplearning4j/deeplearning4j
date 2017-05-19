package org.deeplearning4j.eval;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.transforms.Not;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.shade.jackson.annotation.JsonIgnore;

import java.io.Serializable;
import java.util.*;

/**
 * ROC (Receiver Operating Characteristic) for multi-task binary classifiers, using the specified number of threshold steps.
 * <p>
 * Some ROC implementations will automatically calculate the threshold points based on the data set to give a 'smoother'
 * ROC curve (or optimal cut points for diagnostic purposes). This implementation currently uses fixed steps of size
 * 1.0 / thresholdSteps, as this allows easy implementation for batched and distributed evaluation scenarios (where the
 * full data set is not available in memory on any one machine at once).
 * <p>
 * Unlike {@link ROC} (which supports a single binary label (as a single column probability, or 2 column 'softmax' probability
 * distribution), ROCBinary assumes that all outputs are independent binary variables. This also differs from
 * {@link ROCMultiClass}, which should be used for multi-class (single non-binary) cases.
 * <p>
 * ROCBinary supports per-example and per-output masking: for per-output masking, any particular output may be absent
 * (mask value 0) and hence won't be included in the calculated ROC.
 */
@EqualsAndHashCode(callSuper = true)
@Data
@NoArgsConstructor
public class ROCBinary extends BaseEvaluation<ROCBinary> {
    public static final int DEFAULT_PRECISION = 4;

    private  int thresholdSteps;
    private long[] countActualPositive;
    private long[] countActualNegative;
    private Map<Double, CountsForThreshold> countsForThresholdMap;
    private List<String> labels;

    public ROCBinary(int thresholdSteps) {
        this.thresholdSteps = thresholdSteps;
        countActualNegative = null;
        countActualPositive = null;
    }


    @Override
    public void reset() {
        countActualPositive = null;
        countActualNegative = null;
        countsForThresholdMap = null;
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions) {
        eval(labels, networkPredictions, (INDArray) null);
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions, INDArray maskArray) {
        if (countActualPositive != null && countActualPositive.length != labels.size(1)) {
            throw new IllegalStateException("Labels array does not match stored state size. Expected labels array with "
                            + "size " + countActualPositive.length + ", got labels array with size " + labels.size(1));
        }

        if (labels.rank() == 3) {
            evalTimeSeries(labels, networkPredictions, maskArray);
            return;
        }

        if (countActualPositive == null) {
            //Initialize
            countActualPositive = new long[labels.size(1)];
            countActualNegative = new long[labels.size(1)];

            countsForThresholdMap = new LinkedHashMap<>();
            double step = 1.0 / thresholdSteps;
            for (int i = 0; i <= thresholdSteps; i++) {
                double currThreshold = i * step;
                countsForThresholdMap.put(currThreshold, new CountsForThreshold(currThreshold, labels.size(1)));
            }
        }

        //First: need to increment actual positive/negative (label counts) for each output
        INDArray actual1 = labels;
        INDArray actual0 = Nd4j.getExecutioner().execAndReturn(new Not(labels.dup()));
        if (maskArray != null) {
            actual1 = actual1.mul(maskArray);
            actual0.muli(maskArray);
        }

        int[] countActualPosThisBatch = actual1.sum(0).data().asInt();
        int[] countActualNegThisBatch = actual0.sum(0).data().asInt();
        addInPlace(countActualPositive, countActualPosThisBatch);
        addInPlace(countActualNegative, countActualNegThisBatch);

        //Here: calculate true positive rate (TPR) vs. false positive rate (FPR) at different threshold
        double step = 1.0 / thresholdSteps;
        for (int i = 0; i <= thresholdSteps; i++) {
            double currThreshold = i * step;

            //Work out true/false positives - do this by replacing probabilities (predictions) with 1 or 0 based on threshold
            Condition condGeq = Conditions.greaterThanOrEqual(currThreshold);
            Condition condLeq = Conditions.lessThanOrEqual(currThreshold);

            Op op = new CompareAndSet(networkPredictions.dup(), 1.0, condGeq);
            INDArray predictedClass1 = Nd4j.getExecutioner().execAndReturn(op);
            op = new CompareAndSet(predictedClass1, 0.0, condLeq);
            predictedClass1 = Nd4j.getExecutioner().execAndReturn(op);

            //True positives: occur whet the predicted and actual are both 1s
            //False positives: occur when predicted 1, actual is 0
            INDArray isTruePositive = predictedClass1.mul(actual1);
            INDArray isFalsePositive = predictedClass1.mul(actual0);

            //Apply mask array:
            if (maskArray != null) {
                if (Arrays.equals(labels.shape(), maskArray.shape())) {
                    //Per output masking
                    isTruePositive.muli(maskArray);
                    isFalsePositive.muli(maskArray);
                } else {
                    //Per-example masking
                    isTruePositive.muliColumnVector(maskArray);
                    isFalsePositive.muliColumnVector(maskArray);
                }
            }

            //TP/FP counts for this threshold
            int[] truePositiveCount = isTruePositive.sum(0).data().asInt();
            int[] falsePositiveCount = isFalsePositive.sum(0).data().asInt();

            CountsForThreshold cft = countsForThresholdMap.get(currThreshold);
            cft.incrementTruePositive(truePositiveCount);
            cft.incrementFalsePositive(falsePositiveCount);
        }

    }

    private static void addInPlace(long[] addTo, int[] toAdd) {
        for (int i = 0; i < addTo.length; i++) {
            addTo[i] += toAdd[i];
        }
    }

    private static void addInPlace(long[] addTo, long[] toAdd) {
        for (int i = 0; i < addTo.length; i++) {
            addTo[i] += toAdd[i];
        }
    }

    @Override
    public void merge(ROCBinary other) {
        if (this.countActualPositive == null) {
            this.countActualPositive = other.countActualPositive;
            this.countActualNegative = other.countActualNegative;
            this.countsForThresholdMap = other.countsForThresholdMap;
            return;
        } else if (other.countActualPositive == null) {
            return;
        }

        if (this.countActualPositive.length != other.countActualPositive.length) {
            throw new IllegalStateException("Cannot merge ROCBinary instances with different number of coulmns. "
                            + "numColumns = " + this.countActualPositive.length + "; other numColumns = "
                            + other.countActualPositive.length);
        }

        //Both have data
        addInPlace(this.countActualPositive, other.countActualPositive);
        addInPlace(this.countActualNegative, other.countActualNegative);
        for (Map.Entry<Double, CountsForThreshold> e : countsForThresholdMap.entrySet()) {
            CountsForThreshold o = other.countsForThresholdMap.get(e.getKey());

            e.getValue().incrementTruePositive(o.getCountTruePositive());
            e.getValue().incrementFalsePositive(o.getCountFalsePositive());
        }
    }

    private void assertIndex(int outputNum) {
        if (countActualPositive == null) {
            throw new UnsupportedOperationException("ROCBinary does not have any stats: eval must be called first");
        }
        if (outputNum < 0 || outputNum >= countActualPositive.length) {
            throw new IllegalArgumentException("Invalid input: output number must be between 0 and " + (outputNum - 1));
        }
    }

    /**
     * Returns the number of labels - (i.e., size of the prediction/labels arrays) - if known. Returns -1 otherwise
     */
    public int numLabels() {
        if (countActualPositive == null) {
            return -1;
        }

        return countActualPositive.length;
    }

    /**
     * Get the actual positive count (accounting for any masking) for  the specified output/column
     *
     * @param outputNum Index of the output (0 to {@link #numLabels()}-1)
     */
    public long getCountActualPositive(int outputNum) {
        assertIndex(outputNum);
        return countActualPositive[outputNum];
    }

    /**
     * Get the actual negative count (accounting for any masking) for  the specified output/column
     *
     * @param outputNum Index of the output (0 to {@link #numLabels()}-1)
     */
    public long getCountActualNegative(int outputNum) {
        assertIndex(outputNum);
        return countActualNegative[outputNum];
    }

    /**
     * Get the ROC curve, as a set of points
     *
     * @param outputNum Index of the output (0 to {@link #numLabels()}-1)
     * @return ROC curve, as a list of points
     */
    public List<ROCBinary.ROCValue> getResults(int outputNum) {
        assertIndex(outputNum);
        List<ROCBinary.ROCValue> out = new ArrayList<>(countsForThresholdMap.size());

        for (Map.Entry<Double, ROCBinary.CountsForThreshold> entry : countsForThresholdMap.entrySet()) {
            double t = entry.getKey();
            ROCBinary.CountsForThreshold c = entry.getValue();
            double tpr = c.getCountTruePositive()[outputNum] / ((double) countActualPositive[outputNum]);
            double fpr = c.getCountFalsePositive()[outputNum] / ((double) countActualNegative[outputNum]);

            out.add(new ROCBinary.ROCValue(t, tpr, fpr));
        }

        return out;
    }

    /**
     * Get the precision/recall curve, for the specified output
     *
     * @param outputNum Index of the output (0 to {@link #numLabels()}-1)
     * @return the precision/recall curve
     */
    public List<ROCBinary.PrecisionRecallPoint> getPrecisionRecallCurve(int outputNum) {
        assertIndex(outputNum);
        //Precision: (true positive count) / (true positive count + false positive count) == true positive rate
        //Recall: (true positive count) / (true positive count + false negative count) = (TP count) / (total dataset positives)

        List<ROCBinary.PrecisionRecallPoint> out = new ArrayList<>(countsForThresholdMap.size());

        for (Map.Entry<Double, ROCBinary.CountsForThreshold> entry : countsForThresholdMap.entrySet()) {
            double t = entry.getKey();
            ROCBinary.CountsForThreshold c = entry.getValue();
            long tpCount = c.getCountTruePositive()[outputNum];
            long fpCount = c.getCountFalsePositive()[outputNum];
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
            if (countActualPositive[outputNum] == 0) {
                recall = 1.0;
            } else {
                recall = tpCount / ((double) countActualPositive[outputNum]);
            }


            out.add(new ROCBinary.PrecisionRecallPoint(c.getThreshold(), precision, recall));
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
    public double[][] getResultsAsArray(int outputNum) {
        assertIndex(outputNum);

        double[][] out = new double[2][thresholdSteps + 1];
        int i = 0;
        for (Map.Entry<Double, ROCBinary.CountsForThreshold> entry : countsForThresholdMap.entrySet()) {
            ROCBinary.CountsForThreshold c = entry.getValue();
            double tpr = c.getCountTruePositive()[outputNum] / ((double) countActualPositive[outputNum]);
            double fpr = c.getCountFalsePositive()[outputNum] / ((double) countActualNegative[outputNum]);

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
     * @param outputNum
     * @return AUC
     */
    public double calculateAUC(int outputNum) {
        assertIndex(outputNum);

        //Calculate AUC using trapezoidal rule
        List<ROCBinary.ROCValue> list = getResults(outputNum);

        //Given the points
        double auc = 0.0;
        for (int i = 0; i < list.size() - 1; i++) {
            ROCBinary.ROCValue left = list.get(i);
            ROCBinary.ROCValue right = list.get(i + 1);

            //y axis: TPR
            //x axis: FPR
            double deltaX = Math.abs(right.getFalsePositiveRate() - left.getFalsePositiveRate()); //Iterating in threshold order, so FPR decreases as threshold increases
            double avg = (left.getTruePositiveRate() + right.getTruePositiveRate()) / 2.0;

            auc += deltaX * avg;
        }
        return auc;
    }

    /**
     * Set the label names, for printing via {@link #stats()}
     */
    public void setLabelNames(List<String> labels) {
        if (labels == null) {
            this.labels = null;
            return;
        }
        this.labels = new ArrayList<>(labels);
    }

    @Override
    public String stats() {
        return stats(DEFAULT_PRECISION);
    }

    public String stats(int printPrecision) {
        //Calculate AUC and also print counts, for each output

        StringBuilder sb = new StringBuilder();

        int maxLabelsLength = 15;
        if (labels != null) {
            for (String s : labels) {
                maxLabelsLength = Math.max(s.length(), maxLabelsLength);
            }
        }

        String patternHeader = "%-" + (maxLabelsLength + 5) + "s%-12s%-10s%-10s";
        String header = String.format(patternHeader, "Label", "AUC", "# Pos", "# Neg");

        String pattern = "%-" + (maxLabelsLength + 5) + "s" //Label
                        + "%-12." + printPrecision + "f" //AUC
                        + "%-10d%-10d"; //Count pos, count neg

        sb.append(header);

        for (int i = 0; i < countActualPositive.length; i++) {
            double auc = calculateAUC(i);

            String label = (labels == null ? String.valueOf(i) : labels.get(i));

            sb.append("\n").append(String.format(pattern, label, auc, countActualPositive[i], countActualNegative[i]));
        }

        return sb.toString();
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
        private long[] countTruePositive;
        private long[] countFalsePositive;

        public CountsForThreshold(double threshold, int size) {
            this(threshold, new long[size], new long[size]);
        }

        public void incrementTruePositive(int[] counts) {
            addInPlace(countTruePositive, counts);
        }

        public void incrementFalsePositive(int[] counts) {
            addInPlace(countFalsePositive, counts);
        }

        public void incrementTruePositive(long[] counts) {
            addInPlace(countTruePositive, counts);
        }

        public void incrementFalsePositive(long[] counts) {
            addInPlace(countFalsePositive, counts);
        }

        public void incrementTruePositive(long count, int index) {
            countTruePositive[index] += count;
        }

        public void incrementFalsePositive(long count, int index) {
            countFalsePositive[index] += count;
        }

        @Override
        public ROCBinary.CountsForThreshold clone() {
            long[] ctp = ArrayUtils.clone(countTruePositive);
            long[] tfp = ArrayUtils.clone(countFalsePositive);
            return new ROCBinary.CountsForThreshold(threshold, ctp, tfp);
        }
    }
}
