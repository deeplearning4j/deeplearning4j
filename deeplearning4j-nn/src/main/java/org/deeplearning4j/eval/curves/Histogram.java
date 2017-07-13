package org.deeplearning4j.eval.curves;

import lombok.Data;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * A simple histogram used in evaluation classes, such as {@link org.deeplearning4j.eval.EvaluationCalibration}
 *
 * @author Alex Black
 */
@Data
public class Histogram extends BaseHistogram {
    private final String title;
    private final double lower;
    private final double upper;
    private final int[] binCounts;

    public Histogram(@JsonProperty("title") String title, @JsonProperty("lower") double lower,
                    @JsonProperty("upper") double upper, @JsonProperty("binCounts") int[] binCounts) {
        this.title = title;
        this.lower = lower;
        this.upper = upper;
        this.binCounts = binCounts;
    }

    @Override
    public int numPoints() {
        return binCounts.length;
    }

    @Override
    public double[] getBinLowerBounds() {
        double step = 1.0 / binCounts.length;
        double[] out = new double[binCounts.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = i * step;
        }
        return out;
    }

    @Override
    public double[] getBinUpperBounds() {
        double step = 1.0 / binCounts.length;
        double[] out = new double[binCounts.length];
        for (int i = 0; i < out.length - 1; i++) {
            out[i] = (i + 1) * step;
        }
        out[out.length - 1] = 1.0;
        return out;
    }

    @Override
    public double[] getBinMidValues() {
        double step = 1.0 / binCounts.length;
        double[] out = new double[binCounts.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = (i + 0.5) * step;
        }
        return out;
    }

    /**
     * @param json       JSON representation
     * @return           Instance of the histogram
     */
    public static Histogram fromJson(String json) {
        return BaseHistogram.fromJson(json, Histogram.class);
    }

    /**
     *
     * @param yaml       YAML representation
     * @return           Instance of the histogram
     */
    public static Histogram fromYaml(String yaml) {
        return BaseHistogram.fromYaml(yaml, Histogram.class);
    }
}
