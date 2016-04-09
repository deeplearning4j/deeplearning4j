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

    //No arg constructor for Jackson
    public ChartHistogram(){
        super(COMPONENT_TYPE);
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

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        sb.append("ChartHistogram(lowerBounds=");
        if(lowerBounds != null){
            sb.append(lowerBounds);
        } else {
            sb.append("[]");
        }
        sb.append(",upperBounds=");
        if(upperBounds!= null){
            sb.append(upperBounds);
        } else {
            sb.append("[]");
        }
        sb.append(",yValues=");
        if(yValues != null){
            sb.append(yValues);
        } else {
            sb.append("[]");
        }

        sb.append(")");
        return sb.toString();
    }
}
