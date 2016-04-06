package org.deeplearning4j.ui.components.chart;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.ui.components.chart.style.StyleChart;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Alex on 15/03/2016.
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class ChartStackedArea extends Chart {

    public static final String COMPONENT_TYPE = "ChartStackedArea";

    private String title;
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

        public Builder setXValues(double[] x) {
            this.x = x;
            return this;
        }

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
