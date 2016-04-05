package org.deeplearning4j.ui.components.chart.style;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.ui.api.Style;
import org.deeplearning4j.ui.api.Utils;

import java.awt.*;

/**
 * Created by Alex on 3/04/2016.
 */
@AllArgsConstructor @Data
public class ChartStyle extends Style {

    public static final Double DEFAULT_CHART_MARGIN_TOP = 60.0;
    public static final Double DEFAULT_CHART_MARGIN_BOTTOM = 20.0;
    public static final Double DEFAULT_CHART_MARGIN_LEFT = 60.0;
    public static final Double DEFAULT_CHART_MARGIN_RIGHT = 20.0;

    protected double strokeWidth;
    protected String[] seriesColors;
    protected Double axisStrokeWidth;

    private ChartStyle(Builder b){
        super(b);
        this.strokeWidth = b.strokeWidth;
        this.seriesColors = b.seriesColors;
        this.axisStrokeWidth = b.axisStrokeWidth;
    }



    public static class Builder extends Style.Builder<Builder>{

        protected double strokeWidth = 1.0;
        protected String[] seriesColors;
        protected Double axisStrokeWidth;

        public Builder(){
            super.marginTop = DEFAULT_CHART_MARGIN_TOP;
            super.marginBottom = DEFAULT_CHART_MARGIN_BOTTOM;
            super.marginLeft = DEFAULT_CHART_MARGIN_LEFT;
            super.marginRight = DEFAULT_CHART_MARGIN_RIGHT;
        }

        public Builder strokeWidth(double strokeWidth){
            this.strokeWidth = strokeWidth;
            return this;
        }

        public Builder seriesColors(Color... colors){
            String[] str = new String[colors.length];
            for( int i=0; i<str.length; i++ ) str[i] = Utils.colorToHex(colors[i]);
            return seriesColors(str);
        }

        public Builder seriesColors(String... colors){
            this.seriesColors = colors;
            return this;
        }

        public Builder axisStrokeWidth(double axisStrokeWidth){
            this.axisStrokeWidth = axisStrokeWidth;
            return this;
        }

        public ChartStyle build(){
            return new ChartStyle(this);
        }
    }

}
