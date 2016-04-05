package org.deeplearning4j.ui.components.chart;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.ui.components.chart.style.ChartStyle;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Alex on 25/03/2016.
 */
@EqualsAndHashCode(callSuper = true)
@Data
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

        public Builder(String title, ChartStyle style) {
            super(title, style);
        }

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
