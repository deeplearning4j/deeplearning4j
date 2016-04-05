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

import io.skymind.ui.components.chart.style.ChartStyle;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.util.ArrayList;
import java.util.List;

@EqualsAndHashCode(callSuper = true)
@Data
public class ChartHorizontalBar extends Chart {
    public static final String COMPONENT_TYPE = "ChartHorizontalBar";

    private List<String> labels = new ArrayList<>();
    private List<Double> values = new ArrayList<>();
    private Double xmin;
    private Double xmax;

    private ChartHorizontalBar(Builder builder) {
        super(COMPONENT_TYPE, builder);
        labels = builder.labels;
        values = builder.values;
        this.xmin = builder.xMin;
        this.xmax = builder.xMax;
    }

    public ChartHorizontalBar() {
        super(COMPONENT_TYPE, null);
        //no-arg constructor for Jackson
    }


    public static class Builder extends Chart.Builder<Builder> {

        private List<String> labels = new ArrayList<>();
        private List<Double> values = new ArrayList<>();
        private Double xMin;
        private Double xMax;

        public Builder(String title, ChartStyle style) {
            super(title, style);
        }

        public Builder addValue(String name, double value) {
            labels.add(name);
            values.add(value);
            return this;
        }

        public Builder addValues(List<String> names, double[] values) {
            for (int i = 0; i < names.size(); i++) {
                addValue(names.get(i), values[i]);
            }
            return this;
        }

//        public Builder margins(int top, int bottom, int left, int right){
//            this.marginTop = top;
//            this.marginBottom = bottom;
//            this.marginLeft = left;
//            this.marginRight = right;
//            return this;
//        }

        public Builder xMin(double xMin) {
            this.xMin = xMin;
            return this;
        }

        public Builder xMax(double xMax) {
            this.xMax = xMax;
            return this;
        }

        public Builder addValues(List<String> names, float[] values) {
            for (int i = 0; i < names.size(); i++) {
                addValue(names.get(i), values[i]);
            }
            return this;
        }

        public ChartHorizontalBar build() {
            return new ChartHorizontalBar(this);
        }

    }

}
