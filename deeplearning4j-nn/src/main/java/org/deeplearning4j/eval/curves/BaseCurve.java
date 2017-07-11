package org.deeplearning4j.eval.curves;

import org.deeplearning4j.eval.BaseEvaluation;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.io.IOException;

/**
 * Abstract class for ROC and Precision recall curves
 *
 * @author Alex Black
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY)
public abstract class BaseCurve {
    public static final int DEFAULT_FORMAT_PREC = 4;

    /**
     * @return The number of points in the curve
     */
    public abstract int numPoints();

    /**
     * @return X axis values
     */
    public abstract double[] getX();

    /**
     * @return Y-axis values
     */
    public abstract double[] getY();

    /**
     * @return Title for the curve
     */
    public abstract String getTitle();

    /**
     * @return Area under the curve
     */
    protected double calculateArea() {
        return calculateArea(getX(), getY());
    }

    protected double calculateArea(double[] x, double[] y) {
        int nPoints = x.length;
        double area = 0.0;
        for (int i = 0; i < nPoints - 1; i++) {
            double xLeft = x[i];
            double yLeft = y[i];
            double xRight = x[i + 1];
            double yRight = y[i + 1];

            //y axis: TPR
            //x axis: FPR
            double deltaX = Math.abs(xRight - xLeft); //Iterating in threshold order, so FPR decreases as threshold increases
            double avg = (yRight + yLeft) / 2.0;

            area += deltaX * avg;
        }

        return area;
    }

    protected String format(double d, int precision) {
        return String.format("%." + precision + "f", d);
    }

    /**
     * @return  JSON representation of the curve
     */
    public String toJson() {
        try {
            return BaseEvaluation.getObjectMapper().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * @return YAML  representation of the curve
     */
    public String toYaml() {
        try {
            return BaseEvaluation.getYamlMapper().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     *
     * @param json       JSON representation
     * @param curveClass Class for the curve
     * @param <T>        Type
     * @return           Instance of the curve
     */
    public static <T extends BaseCurve> T fromJson(String json, Class<T> curveClass) {
        try {
            return BaseEvaluation.getObjectMapper().readValue(json, curveClass);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     *
     * @param yaml       YAML representation
     * @param curveClass Class for the curve
     * @param <T>        Type
     * @return           Instance of the curve
     */
    public static <T extends BaseCurve> T fromYaml(String yaml, Class<T> curveClass) {
        try {
            return BaseEvaluation.getYamlMapper().readValue(yaml, curveClass);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
