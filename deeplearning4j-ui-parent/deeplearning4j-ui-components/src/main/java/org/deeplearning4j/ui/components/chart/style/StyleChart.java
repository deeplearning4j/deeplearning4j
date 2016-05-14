/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.ui.components.chart.style;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.ui.api.Style;
import org.deeplearning4j.ui.api.Utils;
import org.deeplearning4j.ui.components.text.style.StyleText;

import java.awt.*;

/**
 * Style for charts
 *
 * @author Alex Black
 */
@AllArgsConstructor @Data @NoArgsConstructor @EqualsAndHashCode(callSuper=true)
@JsonInclude(JsonInclude.Include.NON_NULL)
public class StyleChart extends Style {

    public static final Double DEFAULT_CHART_MARGIN_TOP = 60.0;
    public static final Double DEFAULT_CHART_MARGIN_BOTTOM = 20.0;
    public static final Double DEFAULT_CHART_MARGIN_LEFT = 60.0;
    public static final Double DEFAULT_CHART_MARGIN_RIGHT = 20.0;

    protected Double strokeWidth;
    protected Double pointSize;
    protected String[] seriesColors;
    protected Double axisStrokeWidth;
    protected StyleText titleStyle;

    private StyleChart(Builder b){
        super(b);
        this.strokeWidth = b.strokeWidth;
        this.pointSize = b.pointSize;
        this.seriesColors = b.seriesColors;
        this.axisStrokeWidth = b.axisStrokeWidth;
        this.titleStyle = b.titleStyle;
    }



    public static class Builder extends Style.Builder<Builder>{

        protected Double strokeWidth;
        protected Double pointSize;
        protected String[] seriesColors;
        protected Double axisStrokeWidth;
        protected StyleText titleStyle;

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

        /** Point size, for scatter plot etc */
        public Builder pointSize(double pointSize){
            this.pointSize = pointSize;
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

        public Builder titleStyle(StyleText style){
            this.titleStyle = style;
            return this;
        }

        public StyleChart build(){
            return new StyleChart(this);
        }
    }

}
