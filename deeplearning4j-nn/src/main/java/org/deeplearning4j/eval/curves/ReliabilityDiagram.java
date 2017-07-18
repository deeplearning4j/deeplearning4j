package org.deeplearning4j.eval.curves;

import lombok.Getter;
import lombok.NonNull;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Created by Alex on 05/07/2017.
 */
@Getter
public class ReliabilityDiagram extends BaseCurve {

    private final String title;
    private final double[] meanPredictedValueX;
    private final double[] fractionPositivesY;


    public ReliabilityDiagram(@JsonProperty("title") String title,
                    @NonNull @JsonProperty("meanPredictedValueX") double[] meanPredictedValueX,
                    @NonNull @JsonProperty("fractionPositivesY") double[] fractionPositivesY) {
        this.title = title;
        this.meanPredictedValueX = meanPredictedValueX;
        this.fractionPositivesY = fractionPositivesY;
    }

    @Override
    public int numPoints() {
        return meanPredictedValueX.length;
    }

    @Override
    public double[] getX() {
        return getMeanPredictedValueX();
    }

    @Override
    public double[] getY() {
        return getFractionPositivesY();
    }

    @Override
    public String getTitle() {
        if (title == null) {
            return "Reliability Diagram";
        }
        return title;
    }
}
