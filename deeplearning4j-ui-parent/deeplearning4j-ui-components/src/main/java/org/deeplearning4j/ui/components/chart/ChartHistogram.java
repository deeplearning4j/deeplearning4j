package org.deeplearning4j.ui.components.chart;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.ui.components.chart.style.StyleChart;

import java.util.ArrayList;
import java.util.List;

/**
 * Histogram chart, with pre-binned values. Supports variable width bins
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ChartHistogram extends Chart {
    public static final String COMPONENT_TYPE = "ChartHistogram";

    private List<Double> lowerBounds = new ArrayList<>();
    private List<Double> upperBounds = new ArrayList<>();
    private List<Double> yValues = new ArrayList<>();

    public ChartHistogram(Builder builder) {
        super(COMPONENT_TYPE, builder);
        this.lowerBounds = builder.lowerBounds;
        this.upperBounds = builder.upperBounds;
        this.yValues = builder.yValues;
    }


    public static class Builder extends Chart.Builder<Builder> {
        private List<Double> lowerBounds = new ArrayList<>();
        private List<Double> upperBounds = new ArrayList<>();
        private List<Double> yValues = new ArrayList<>();

        public Builder(String title, StyleChart style) {
            super(title, style);
        }

        /**
         * Add a single bin
         *
         * @param lower  Lower (minimum/left) value for the bin (x axis)
         * @param upper  Upper (maximum/right) value for the bin (x axis)
         * @param yValue The height of the bin
         */
        public Builder addBin(double lower, double upper, double yValue) {
            lowerBounds.add(lower);
            upperBounds.add(upper);
            yValues.add(yValue);
            return this;
        }

        public ChartHistogram build() {
            return new ChartHistogram(this);
        }
    }
}
