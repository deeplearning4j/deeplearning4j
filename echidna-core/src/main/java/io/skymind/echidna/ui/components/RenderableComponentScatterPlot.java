package io.skymind.echidna.ui.components;

import lombok.Data;
import lombok.EqualsAndHashCode;

import java.util.ArrayList;
import java.util.List;

@EqualsAndHashCode(callSuper = true)
@Data
public class RenderableComponentScatterPlot extends RenderableComponent {
    public static final String COMPONENT_TYPE = "scatterplot";

    private RenderableComponentScatterPlot(Builder builder){
        super(COMPONENT_TYPE);
        title = builder.title;
        x = builder.x;
        y = builder.y;
        seriesNames = builder.seriesNames;
    }

    public RenderableComponentScatterPlot(){
        super(COMPONENT_TYPE);
        //no-arg constructor for Jackson
    }

    private String title;
    private List<double[]> x;
    private List<double[]> y;
    private List<String> seriesNames;



    public static class Builder {

        private String title;
        private List<double[]> x = new ArrayList<>();
        private List<double[]> y = new ArrayList<>();
        private List<String> seriesNames = new ArrayList<>();

        public Builder title(String title){
            this.title = title;
            return this;
        }

        public Builder addSeries(String seriesName, double[] xValues, double[] yValues){
            x.add(xValues);
            y.add(yValues);
            seriesNames.add(seriesName);
            return this;
        }

        public RenderableComponentScatterPlot build(){
            return new RenderableComponentScatterPlot(this);
        }
    }

}
