package org.deeplearning4j.ui.components.chart;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.ui.components.chart.style.StyleChart;

import java.util.ArrayList;
import java.util.List;

/**
 * Stacked area chart (no normalization), with multiple series.
 * Note that in the current implementation, the x values for each series must be the same
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ChartStackedArea extends Chart {

    public static final String COMPONENT_TYPE = "ChartStackedArea";

    private double[] x = new double[0];
    private List<double[]> y = new ArrayList<>();
    private List<String> labels = new ArrayList<>();

    public ChartStackedArea() {
        super(COMPONENT_TYPE, null);
    }

    public ChartStackedArea(Builder builder) {
        super(COMPONENT_TYPE, builder);
        this.x = builder.x;
        this.y = builder.y;
        this.labels = builder.seriesNames;
    }

    public static class Builder extends Chart.Builder<Builder> {

        private double[] x;
        private List<double[]> y = new ArrayList<>();
        private List<String> seriesNames = new ArrayList<>();


        public Builder(String title, StyleChart style) {
            super(title, style);
        }

        /**
         * Set the x-axis values
         */
        public Builder setXValues(double[] x) {
            this.x = x;
            return this;
        }

        /**
         * Add a single series.
         *
         * @param seriesName Name of the series
         * @param yValues    length of the yValues array must be same as the x-values array
         */
        public Builder addSeries(String seriesName, double[] yValues) {
            y.add(yValues);
            seriesNames.add(seriesName);
            return this;
        }

        public ChartStackedArea build() {
            return new ChartStackedArea(this);
        }
    }
}
