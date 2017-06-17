package org.deeplearning4j.eval.curves;

import com.google.common.base.Preconditions;
import lombok.Data;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Created by Alex on 17/06/2017.
 */
@Data
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

    public double getThreshold(int i){
        Preconditions.checkArgument(i >= 0 && i < threshold.length, "Invalid index: " + i);
        return threshold[i];
    }

    public double getPrecision(int i){
        Preconditions.checkArgument(i >= 0 && i < precision.length, "Invalid index: " + i);
        return precision[i];
    }

    public double getRecall(int i){
        Preconditions.checkArgument(i >= 0 && i < recall.length, "Invalid index: " + i);
        return recall[i];
    }

    public double calculateAUPRC(){
        if(area != null){
            return area;
        }

        area = calculateArea();
        return area;
    }

}
