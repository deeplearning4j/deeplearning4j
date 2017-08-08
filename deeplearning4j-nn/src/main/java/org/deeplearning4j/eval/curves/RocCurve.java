package org.deeplearning4j.eval.curves;

import com.google.common.base.Preconditions;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * ROC curve: a set of (false positive, true positive) tuples at different thresholds
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(exclude = {"auc"}, callSuper = false)
public class RocCurve extends BaseCurve {

    private double[] threshold;
    private double[] fpr;
    private double[] tpr;

    private Double auc;

    public RocCurve(@JsonProperty("threshold") double[] threshold, @JsonProperty("fpr") double[] fpr,
                    @JsonProperty("tpr") double[] tpr) {
        this.threshold = threshold;
        this.fpr = fpr;
        this.tpr = tpr;
    }


    @Override
    public int numPoints() {
        return threshold.length;
    }

    @Override
    public double[] getX() {
        return fpr;
    }

    @Override
    public double[] getY() {
        return tpr;
    }

    @Override
    public String getTitle() {
        return "ROC (Area=" + format(calculateAUC(), DEFAULT_FORMAT_PREC) + ")";
    }

    /**
     * @param i Point number, 0 to numPoints()-1 inclusive
     * @return Threshold of a given point
     */
    public double getThreshold(int i) {
        Preconditions.checkArgument(i >= 0 && i < threshold.length, "Invalid index: " + i);
        return threshold[i];
    }

    /**
     * @param i Point number, 0 to numPoints()-1 inclusive
     * @return True positive rate of a given point
     */
    public double getTruePositiveRate(int i) {
        Preconditions.checkArgument(i >= 0 && i < tpr.length, "Invalid index: " + i);
        return tpr[i];
    }

    /**
     * @param i Point number, 0 to numPoints()-1 inclusive
     * @return False positive rate of a given point
     */
    public double getFalsePositiveRate(int i) {
        Preconditions.checkArgument(i >= 0 && i < fpr.length, "Invalid index: " + i);
        return fpr[i];
    }

    /**
     * Calculate and return the area under ROC curve
     */
    public double calculateAUC() {
        if (auc != null) {
            return auc;
        }

        auc = calculateArea();
        return auc;
    }

    public static RocCurve fromJson(String json) {
        return fromJson(json, RocCurve.class);
    }

    public static RocCurve fromYaml(String yaml) {
        return fromYaml(yaml, RocCurve.class);
    }

}
