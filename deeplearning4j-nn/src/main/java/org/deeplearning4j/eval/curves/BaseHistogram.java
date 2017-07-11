package org.deeplearning4j.eval.curves;

import org.deeplearning4j.eval.BaseEvaluation;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.io.IOException;

/**
 * Created by Alex on 06/07/2017.
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY)
public abstract class BaseHistogram {

    public abstract String getTitle();

    public abstract int numPoints();

    public abstract int[] getBinCounts();

    public abstract double[] getBinLowerBounds();

    public abstract double[] getBinUpperBounds();

    public abstract double[] getBinMidValues();


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
    public static <T extends BaseHistogram> T fromJson(String json, Class<T> curveClass) {
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
    public static <T extends BaseHistogram> T fromYaml(String yaml, Class<T> curveClass) {
        try {
            return BaseEvaluation.getYamlMapper().readValue(yaml, curveClass);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
