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
import lombok.Getter;
import org.deeplearning4j.ui.api.Component;
import org.deeplearning4j.ui.components.chart.style.StyleChart;
import org.nd4j.shade.jackson.annotation.JsonInclude;

/**
 * Abstract class for charts
 *
 * @author Alex BLack
 */
@Data
@EqualsAndHashCode(callSuper = true)
@JsonInclude(JsonInclude.Include.NON_NULL)
public abstract class Chart extends Component {

    private String title;
    private Boolean suppressAxisHorizontal;
    private Boolean suppressAxisVertical;
    private boolean showLegend;

    private Double setXMin;
    private Double setXMax;
    private Double setYMin;
    private Double setYMax;

    private Double gridVerticalStrokeWidth;
    private Double gridHorizontalStrokeWidth;

    public Chart(String componentType) {
        super(componentType, null);
    }

    public Chart(String componentType, Builder builder) {
        super(componentType, builder.getStyle());
        this.title = builder.title;
        this.suppressAxisHorizontal = builder.suppressAxisHorizontal;
        this.suppressAxisVertical = builder.suppressAxisVertical;
        this.showLegend = builder.showLegend;

        this.setXMin = builder.setXMin;
        this.setXMax = builder.setXMax;
        this.setYMin = builder.setYMin;
        this.setYMax = builder.setYMax;

        this.gridVerticalStrokeWidth = builder.gridVerticalStrokeWidth;
        this.gridHorizontalStrokeWidth = builder.gridHorizontalStrokeWidth;
    }


    @Getter
    @SuppressWarnings("unchecked")
    public static abstract class Builder<T extends Builder<T>> {

        private String title;
        private StyleChart style;
        private Boolean suppressAxisHorizontal;
        private Boolean suppressAxisVertical;
        private boolean showLegend;

        private Double setXMin;
        private Double setXMax;
        private Double setYMin;
        private Double setYMax;

        private Double gridVerticalStrokeWidth;
        private Double gridHorizontalStrokeWidth;

        /**
         * @param title Title for the chart (may be null)
         * @param style Style for the chart (may be null)
         */
        public Builder(String title, StyleChart style) {
            this.title = title;
            this.style = style;
        }

        /**
         * @param suppressAxisHorizontal if true: don't show the horizontal axis (default: show)
         */
        public T suppressAxisHorizontal(boolean suppressAxisHorizontal) {
            this.suppressAxisHorizontal = suppressAxisHorizontal;
            return (T) this;
        }

        /**
         * @param suppressAxisVertical if true: don't show the vertical axis (default: show)
         */
        public T suppressAxisVertical(boolean suppressAxisVertical) {
            this.suppressAxisVertical = suppressAxisVertical;
            return (T) this;
        }

        /**
         * @param showLegend if true: show the legend. (default: no legend)
         */
        public T showLegend(boolean showLegend) {
            this.showLegend = showLegend;
            return (T) this;
        }

        /**
         * Used to override/set the minimum value for the x axis. If this is not set, x axis minimum is set based on the data
         * @param xMin Minimum value to use for the x axis
         */
        public T setXMin(Double xMin) {
            this.setXMin = xMin;
            return (T) this;
        }

        /**
         * Used to override/set the maximum value for the x axis. If this is not set, x axis maximum is set based on the data
         * @param xMax Maximum value to use for the x axis
         */
        public T setXMax(Double xMax) {
            this.setXMax = xMax;
            return (T) this;
        }

        /**
         * Used to override/set the minimum value for the y axis. If this is not set, y axis minimum is set based on the data
         * @param yMin Minimum value to use for the y axis
         */
        public T setYMin(Double yMin) {
            this.setYMin = yMin;
            return (T) this;
        }

        /**
         * Used to override/set the maximum value for the y axis. If this is not set, y axis minimum is set based on the data
         * @param yMax Minimum value to use for the y axis
         */
        public T setYMax(Double yMax) {
            this.setYMax = yMax;
            return (T) this;
        }

        /**
         * Set the grid lines to be enabled, and if enabled: set the grid.
         * @param gridVerticalStrokeWidth      If null (or 0): show no vertical grid. Otherwise: width in px
         * @param gridHorizontalStrokeWidth    If null (or 0): show no horizontal grid. Otherwise: width in px
         */
        public T setGridWidth(Double gridVerticalStrokeWidth, Double gridHorizontalStrokeWidth) {
            this.gridVerticalStrokeWidth = gridVerticalStrokeWidth;
            this.gridHorizontalStrokeWidth = gridHorizontalStrokeWidth;
            return (T) this;
        }

        /**
         * Set the grid lines to be enabled, and if enabled: set the grid.
         * @param gridVerticalStrokeWidth      If null (or 0): show no vertical grid. Otherwise: width in px
         * @param gridHorizontalStrokeWidth    If null (or 0): show no horizontal grid. Otherwise: width in px
         */
        public T setGridWidth(Integer gridVerticalStrokeWidth, Integer gridHorizontalStrokeWidth) {
            return setGridWidth((gridVerticalStrokeWidth != null ? gridVerticalStrokeWidth.doubleValue() : null),
                            (gridHorizontalStrokeWidth != null ? gridHorizontalStrokeWidth.doubleValue() : null));
        }

    }

}
