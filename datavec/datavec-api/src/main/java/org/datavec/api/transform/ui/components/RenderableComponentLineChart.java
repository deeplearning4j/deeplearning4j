/*-
 *  * Copyright 2016 Skymind, Inc.
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
 */

package org.datavec.api.transform.ui.components;

import lombok.Data;
import lombok.EqualsAndHashCode;

import java.util.ArrayList;
import java.util.List;

@EqualsAndHashCode(callSuper = true)
@Data
public class RenderableComponentLineChart extends RenderableComponent {
    public static final String COMPONENT_TYPE = "linechart";

    private String title;
    private List<double[]> x;
    private List<double[]> y;
    private List<String> seriesNames;
    private boolean removeAxisHorizontal;
    private int marginTop;
    private int marginBottom;
    private int marginLeft;
    private int marginRight;
    private boolean legend;

    private RenderableComponentLineChart(Builder builder) {
        super(COMPONENT_TYPE);
        title = builder.title;
        x = builder.x;
        y = builder.y;
        seriesNames = builder.seriesNames;
        this.removeAxisHorizontal = builder.removeAxisHorizontal;
        this.marginTop = builder.marginTop;
        this.marginBottom = builder.marginBottom;
        this.marginLeft = builder.marginLeft;
        this.marginRight = builder.marginRight;
    }

    public RenderableComponentLineChart() {
        super(COMPONENT_TYPE);
        //no-arg constructor for Jackson
    }



    public static class Builder {

        private String title;
        private List<double[]> x = new ArrayList<>();
        private List<double[]> y = new ArrayList<>();
        private List<String> seriesNames = new ArrayList<>();
        private boolean removeAxisHorizontal = false;
        private boolean legend = true;

        private int marginTop = 60;
        private int marginBottom = 60;
        private int marginLeft = 60;
        private int marginRight = 20;

        public Builder title(String title) {
            this.title = title;
            return this;
        }

        public Builder addSeries(String seriesName, double[] xValues, double[] yValues) {
            x.add(xValues);
            y.add(yValues);
            seriesNames.add(seriesName);
            return this;
        }

        public Builder setRemoveAxisHorizontal(boolean removeAxisHorizontal) {
            this.removeAxisHorizontal = removeAxisHorizontal;
            return this;
        }

        public Builder margins(int top, int bottom, int left, int right) {
            this.marginTop = top;
            this.marginBottom = bottom;
            this.marginLeft = left;
            this.marginRight = right;
            return this;
        }

        public Builder legend(boolean legend) {
            this.legend = legend;
            return this;
        }

        public RenderableComponentLineChart build() {
            return new RenderableComponentLineChart(this);
        }

    }

}
