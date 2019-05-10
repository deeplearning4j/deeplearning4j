/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.ui.components.chart;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.ui.components.chart.style.StyleChart;
import org.nd4j.shade.jackson.annotation.JsonInclude;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Stacked area chart (no normalization), with multiple series.
 * Note that in the current implementation, the x values for each series must be the same
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ChartStackedArea extends Chart {

    public static final String COMPONENT_TYPE = "ChartStackedArea";

    private double[] x = new double[0];
    private List<double[]> y = new ArrayList<>();
    private List<String> labels = new ArrayList<>();

    public ChartStackedArea() {
        super(COMPONENT_TYPE);
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

        /**
         * Set the x-axis values
         */
        public Builder setXValues(double[] x) {
            this.x = x;
            return this;
        }

        /**
         * Add a single series.
         *
         * @param seriesName Name of the series
         * @param yValues    length of the yValues array must be same as the x-values array
         */
        public Builder addSeries(String seriesName, double[] yValues) {
            y.add(yValues);
            seriesNames.add(seriesName);
            return this;
        }

        public ChartStackedArea build() {
            return new ChartStackedArea(this);
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("ChartStackedArea(x=");
        if (x != null) {
            sb.append(Arrays.toString(x));
        } else {
            sb.append("[]");
        }
        sb.append(",y=[");
        boolean first = true;
        if (y != null) {
            for (double[] d : y) {
                if (!first)
                    sb.append(",");
                sb.append(Arrays.toString(d));
                first = false;
            }
        }
        sb.append("],labels=");
        if (labels != null)
            sb.append(labels);
        sb.append(")");
        return sb.toString();
    }
}
