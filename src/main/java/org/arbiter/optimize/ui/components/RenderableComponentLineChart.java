package org.arbiter.optimize.ui.components;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class RenderableComponentLineChart extends RenderableComponent {
    public static final String COMPONENT_TYPE = "linechart";

    private RenderableComponentLineChart(Builder builder){
        super(COMPONENT_TYPE);
        title = builder.title;
        x = null;
        y = null;
    }

    private final String title;
    private final List<double[]> x;
    private final List<double[]> y;



    public static class Builder {

        private String title;
        private List<double[]> x = new ArrayList<>();
        private List<double[]> y = new ArrayList<>();

        public Builder title(String title){
            this.title = title;
            return this;
        }

        public Builder addSeries(String seriesName, double[] xValues, double[] yValues){
            x.add(xValues);
            y.add(yValues);
            //TODO use series name
            return this;
        }


    }

}
