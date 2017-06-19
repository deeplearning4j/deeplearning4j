package org.deeplearning4j.eval.curves;

import com.google.common.base.Preconditions;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Precision recall curve: A set of (recall, precision) points and different thresholds
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(exclude = {"area"}, callSuper = false)
public class PrecisionRecallCurve extends BaseCurve {

    private double[] threshold;
    private double[] precision;
    private double[] recall;

    private Double area;

    public PrecisionRecallCurve(@JsonProperty("threshold") double[] threshold,
                                @JsonProperty("precision") double[] precision,
                                @JsonProperty("recall") double[] recall) {
        this.threshold = threshold;
        this.precision = precision;
        this.recall = recall;
    }

    @Override
    public int numPoints(){
        return threshold.length;
    }

    @Override
    public double[] getX() {
        return recall;
    }

    @Override
    public double[] getY() {
        return precision;
    }

    @Override
    public String getTitle() {
        return "Precision-Recall Curve (Area=" + format(calculateAUPRC(), DEFAULT_FORMAT_PREC) + ")";
    }

    /**
     * @param i Point number, 0 to numPoints()-1 inclusive
     * @return Threshold of a given point
     */
    public double getThreshold(int i){
        Preconditions.checkArgument(i >= 0 && i < threshold.length, "Invalid index: " + i);
        return threshold[i];
    }

    /**
     * @param i Point number, 0 to numPoints()-1 inclusive
     * @return Precision of a given point
     */
    public double getPrecision(int i){
        Preconditions.checkArgument(i >= 0 && i < precision.length, "Invalid index: " + i);
        return precision[i];
    }

    /**
     * @param i Point number, 0 to numPoints()-1 inclusive
     * @return Recall of a given point
     */
    public double getRecall(int i){
        Preconditions.checkArgument(i >= 0 && i < recall.length, "Invalid index: " + i);
        return recall[i];
    }

    /**
     * @return The area under the precision recall curve
     */
    public double calculateAUPRC(){
        if(area != null){
            return area;
        }

        area = calculateArea();
        return area;
    }

    public static PrecisionRecallCurve fromJson(String json){
        return fromJson(json, PrecisionRecallCurve.class);
    }

    public static PrecisionRecallCurve fromYaml(String yaml){
        return fromYaml(yaml, PrecisionRecallCurve.class);
    }

}
