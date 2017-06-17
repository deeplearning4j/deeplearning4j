package org.deeplearning4j.eval.curves;

import com.google.common.base.Preconditions;
import lombok.Data;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Created by Alex on 17/06/2017.
 */
@Data
public class RocCurve extends BaseCurve {

    private double[] threshold;
    private double[] fpr;
    private double[] tpr;

    private Double auc;

    public RocCurve(@JsonProperty("threshold") double[] threshold,
                    @JsonProperty("fpr") double[] fpr,
                    @JsonProperty("tpr") double[] tpr) {
        this.threshold = threshold;
        this.fpr = fpr;
        this.tpr = tpr;
    }


    @Override
    public int numPoints(){
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

    public double getThreshold(int i){
        Preconditions.checkArgument(i >= 0 && i < threshold.length, "Invalid index: " + i);
        return threshold[i];
    }

    public double getTruePositiveRate(int i){
        Preconditions.checkArgument(i >= 0 && i < tpr.length, "Invalid index: " + i);
        return tpr[i];
    }

    public double getFalsePositiveRate(int i){
        Preconditions.checkArgument(i >= 0 && i < fpr.length, "Invalid index: " + i);
        return fpr[i];
    }

    public double calculateAUC(){
        if(auc != null){
            return auc;
        }

        auc = calculateArea();
        return auc;
    }

}
