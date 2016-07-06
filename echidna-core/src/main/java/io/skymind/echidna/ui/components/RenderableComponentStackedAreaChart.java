package io.skymind.echidna.ui.components;

import lombok.Data;
import lombok.EqualsAndHashCode;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Alex on 15/03/2016.
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class RenderableComponentStackedAreaChart extends RenderableComponent {

    public static final String COMPONENT_TYPE = "stackedareachart";

    private String title;
    private double[] x = new double[0];
    private List<double[]> y = new ArrayList<>();
    private List<String> labels = new ArrayList<>();
    private int marginTop;
    private int marginBottom;
    private int marginLeft;
    private int marginRight;
    private boolean removeAxisHorizontal;

    public RenderableComponentStackedAreaChart(){
        super(COMPONENT_TYPE);
    }

    public RenderableComponentStackedAreaChart(Builder builder){
        super(COMPONENT_TYPE);

        this.title = builder.title;
        this.x = builder.x;
        this.y = builder.y;
        this.labels = builder.seriesNames;
        this.marginTop = builder.marginTop;
        this.marginBottom = builder.marginBottom;
        this.marginLeft = builder.marginLeft;
        this.marginRight = builder.marginRight;
        this.removeAxisHorizontal = builder.removeAxisHorizontal;
    }

    public static class Builder {

        private String title;
        private double[] x;
        private List<double[]> y = new ArrayList<>();
        private List<String> seriesNames = new ArrayList<>();

        private int marginTop = 60;
        private int marginBottom = 60;
        private int marginLeft = 60;
        private int marginRight = 20;
        private boolean removeAxisHorizontal = false;

        public Builder title(String title){
            this.title = title;
            return this;
        }

        public Builder setXValues(double[] x){
            this.x = x;
            return this;
        }

        public Builder addSeries(String seriesName, double[] yValues){
            y.add(yValues);
            seriesNames.add(seriesName);
            return this;
        }

        public Builder margins(int top, int bottom, int left, int right){
            this.marginTop = top;
            this.marginBottom = bottom;
            this.marginLeft = left;
            this.marginRight = right;
            return this;
        }

        public Builder setRemoveAxisHorizontal(boolean removeAxisHorizontal){
            this.removeAxisHorizontal = removeAxisHorizontal;
            return this;
        }

        public RenderableComponentStackedAreaChart build(){
            return new RenderableComponentStackedAreaChart(this);
        }

    }

}
