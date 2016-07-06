package io.skymind.echidna.ui.components;

import lombok.Data;
import lombok.EqualsAndHashCode;

import java.util.ArrayList;
import java.util.List;

@EqualsAndHashCode(callSuper = true)
@Data
public class RenderableComponentHorizontalBarChart extends RenderableComponent {
    public static final String COMPONENT_TYPE = "horizontalbarchart";

    private String title;
    private List<String> labels = new ArrayList<>();
    private List<Double> values = new ArrayList<>();
    private int marginTop;
    private int marginBottom;
    private int marginLeft;
    private int marginRight;
    private Double xmin;
    private Double xmax;

    private RenderableComponentHorizontalBarChart(Builder builder){
        super(COMPONENT_TYPE);
        title = builder.title;
        labels = builder.labels;
        values = builder.values;
        this.marginTop = builder.marginTop;
        this.marginBottom = builder.marginBottom;
        this.marginLeft = builder.marginLeft;
        this.marginRight = builder.marginRight;
        this.xmin = builder.xMin;
        this.xmax = builder.xMax;
    }

    public RenderableComponentHorizontalBarChart(){
        super(COMPONENT_TYPE);
        //no-arg constructor for Jackson
    }



    public static class Builder {

        private String title;
        private List<String> labels = new ArrayList<>();
        private List<Double> values = new ArrayList<>();
        private int marginTop = 60;
        private int marginBottom = 60;
        private int marginLeft = 60;
        private int marginRight = 20;
        private Double xMin;
        private Double xMax;

        public Builder title(String title){
            this.title = title;
            return this;
        }

        public Builder addValue(String name, double value){
            labels.add(name);
            values.add(value);
            return this;
        }

        public Builder addValues(List<String> names, double[] values){
            for( int i=0; i<names.size(); i++ ){
                addValue(names.get(i),values[i]);
            }
            return this;
        }

        public Builder margins(int top, int bottom, int left, int right){
            this.marginTop = top;
            this.marginBottom = bottom;
            this.marginLeft = left;
            this.marginRight = right;
            return this;
        }

        public Builder xMin(double xMin){
            this.xMin = xMin;
            return this;
        }

        public Builder xMax(double xMax){
            this.xMax = xMax;
            return this;
        }

        public Builder addValues(List<String> names, float[] values){
            for( int i=0; i<names.size(); i++ ){
                addValue(names.get(i),values[i]);
            }
            return this;
        }

        public RenderableComponentHorizontalBarChart build(){
            return new RenderableComponentHorizontalBarChart(this);
        }

    }

}
