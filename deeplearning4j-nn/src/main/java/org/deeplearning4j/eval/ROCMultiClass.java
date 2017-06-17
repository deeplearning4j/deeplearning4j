package org.deeplearning4j.eval;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.eval.curves.PrecisionRecallCurve;
import org.deeplearning4j.eval.curves.RocCurve;
import org.deeplearning4j.eval.serde.ROCArraySerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import java.util.Arrays;
import java.util.List;

/**
 * ROC (Receiver Operating Characteristic) for multi-class classifiers, using the specified number of threshold steps.
 * <p>
 * The ROC curves are produced by treating the predictions as a set of one-vs-all classifiers, and then calculating
 * ROC curves for each. In practice, this means for N classes, we get N ROC curves.
 * <p>
 * Some ROC implementations will automatically calculate the threshold points based on the data set to give a 'smoother'
 * ROC curve (or optimal cut points for diagnostic purposes). This implementation currently uses fixed steps of size
 * 1.0 / thresholdSteps, as this allows easy implementation for batched and distributed evaluation scenarios (where the
 * full data set is not available in memory on any one machine at once).
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor
public class ROCMultiClass extends BaseEvaluation<ROCMultiClass> {
    public static final int DEFAULT_STATS_PRECISION = 4;

    private int thresholdSteps;
    private boolean rocRemoveRedundantPts;
    @JsonSerialize(using = ROCArraySerializer.class)
    private ROC[] underlying;
    private List<String> labels;

    /**
     * @param thresholdSteps Number of threshold steps to use for the ROC calculation
     */
    public ROCMultiClass(int thresholdSteps) {
        this(thresholdSteps, true);
    }

    public ROCMultiClass(int thresholdSteps, boolean rocRemoveRedundantPts) {
        this.thresholdSteps = thresholdSteps;
        this.rocRemoveRedundantPts = rocRemoveRedundantPts;
    }

    @Override
    public void reset() {
        underlying = null;
    }


    @Override
    public String stats() {
        return stats(DEFAULT_STATS_PRECISION);
    }

    public String stats(int printPrecision){

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

        if(underlying != null) {
            for (int i = 0; i < underlying.length; i++) {
                double auc = calculateAUC(i);

                String label = (labels == null ? String.valueOf(i) : labels.get(i));

                sb.append("\n").append(String.format(pattern, label, auc, getCountActualPositive(i), getCountActualNegative(i)));
            }

            sb.append("Average AUC: ").append(String.format("%-12."+printPrecision+"f", calculateAverageAUC()));
        } else {
            //Empty evaluation
            sb.append("\n-- No Data --\n");
        }

        return sb.toString();
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
        if (labels.rank() > 2 || predictions.rank() > 2 || labels.size(1) != predictions.size(1)) {
            throw new IllegalArgumentException("Invalid input data shape: labels shape = "
                            + Arrays.toString(labels.shape()) + ", predictions shape = "
                            + Arrays.toString(predictions.shape()) + "; require rank 2 array with size(1) == 1 or 2");
        }

        int n = labels.size(1);
        if (underlying == null) {
            underlying = new ROC[n];
            for( int i=0; i<n; i++ ){
                underlying[i] = new ROC(thresholdSteps, rocRemoveRedundantPts);
            }
        }

        if (underlying.length != labels.size(1)) {
            throw new IllegalArgumentException(
                            "Cannot evaluate data: number of label classes does not match previous call. " + "Got "
                                            + labels.size(1) + " labels (from array shape "
                                            + Arrays.toString(labels.shape()) + ")"
                                            + " vs. expected number of label classes = " + underlying.length);
        }

        for( int i=0; i<n; i++ ){
            INDArray prob = predictions.getColumn(i);   //Probability of class i
            INDArray label = labels.getColumn(i);
            underlying[i].eval(label, prob);
        }
    }

    public RocCurve getRocCurve(int classIdx){
        assertIndex(classIdx);
        return underlying[classIdx].getRocCurve();
    }

    public PrecisionRecallCurve getPrecisionRecallCurve(int classIdx){
        assertIndex(classIdx);
        return underlying[classIdx].getPrecisionRecallCurve();
    }

    /**
     * Calculate the AUC - Area Under ROC Curve<br>
     * Utilizes trapezoidal integration internally
     *
     * @return AUC
     */
    public double calculateAUC(int classIdx) {
        assertIndex(classIdx);
        return underlying[classIdx].calculateAUC();
    }

    /**
     * Calculate the AUPRC - Area Under Curve Precision Recall <br>
     * Utilizes trapezoidal integration internally
     *
     * @return AUC
     */
    public double calculateAUCPR(int classIdx) {
        assertIndex(classIdx);
        return underlying[classIdx].calculateAUCPR();
    }

    /**
     * Calculate the macro-average (one-vs-all) AUC for all classes
     */
    public double calculateAverageAUC() {
        assertIndex(0);

        double sum = 0.0;
        for (int i = 0; i < underlying.length; i++) {
            sum += calculateAUC(i);
        }

        return sum / underlying.length;
    }

    /**
     * Get the actual positive count (accounting for any masking) for  the specified class
     *
     * @param outputNum Index of the class
     */
    public long getCountActualPositive(int outputNum) {
        assertIndex(outputNum);
        return underlying[outputNum].getCountActualPositive();
    }

    /**
     * Get the actual negative count (accounting for any masking) for  the specified output/column
     *
     * @param outputNum Index of the class
     */
    public long getCountActualNegative(int outputNum) {
        assertIndex(outputNum);
        return underlying[outputNum].getCountActualNegative();
    }

    /**
     * Merge this ROCMultiClass instance with another.
     * This ROCMultiClass instance is modified, by adding the stats from the other instance.
     *
     * @param other ROCMultiClass instance to combine with this one
     */
    @Override
    public void merge(ROCMultiClass other) {
        if (this.underlying == null) {
            this.underlying = other.underlying;
            return;
        } else if (other.underlying == null) {
            return;
        }

        //Both have data
        if(underlying.length != other.underlying.length){
            throw new UnsupportedOperationException("Cannot merge ROCBinary: this expects " + underlying.length +
                    "outputs, other expects " + other.underlying.length + " outputs");
        }
        for( int i=0; i<underlying.length; i++ ){
            this.underlying[i].merge(other.underlying[i]);
        }
    }

    public int getNumClasses(){
        if(underlying == null){
            return -1;
        }
        return underlying.length;
    }


    private void assertIndex(int classIdx) {
        if (underlying == null) {
            throw new IllegalStateException("Cannot get results: no data has been collected");
        }
        if (classIdx < 0 || classIdx >= underlying.length) {
            throw new IllegalArgumentException("Invalid class index (" + classIdx
                            + "): must be in range 0 to numClasses = " + underlying.length);
        }
    }
}
