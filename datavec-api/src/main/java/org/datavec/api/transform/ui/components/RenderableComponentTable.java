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

@EqualsAndHashCode(callSuper = true)
@Data
public class RenderableComponentTable extends RenderableComponent {

    public static final String COMPONENT_TYPE = "simpletable";

    private String title;
    private String[] header;
    private String[][] table;
    private int padLeftPx = 0;
    private int padRightPx = 0;
    private int padTopPx = 0;
    private int padBottomPx = 0;
    private int border = 0;
    private double[] colWidthsPercent = null;
    private String backgroundColor;
    private String headerColor;

    public RenderableComponentTable() {
        super(COMPONENT_TYPE);
        //No arg constructor for Jackson
    }

    public RenderableComponentTable(Builder builder) {
        super(COMPONENT_TYPE);
        this.title = builder.title;
        this.header = builder.header;
        this.table = builder.table;
        this.padLeftPx = builder.padLeftPx;
        this.padRightPx = builder.padRightPx;
        this.padTopPx = builder.padTopPx;
        this.padBottomPx = builder.padBottomPx;
        this.border = builder.border;
        this.colWidthsPercent = builder.colWidthsPercent;
        this.backgroundColor = builder.backgroundColor;
        this.headerColor = builder.headerColor;
    }

    public RenderableComponentTable(String[] header, String[][] table) {
        this(null, header, table);
    }

    public RenderableComponentTable(String title, String[] header, String[][] table) {
        super(COMPONENT_TYPE);
        this.title = title;
        this.header = header;
        this.table = table;
    }

    public static class Builder {

        private String title;
        private String[] header;
        private String[][] table;
        private int padLeftPx = 0;
        private int padRightPx = 0;
        private int padTopPx = 0;
        private int padBottomPx = 0;
        private int border = 0;
        private double[] colWidthsPercent;
        private String backgroundColor;
        private String headerColor;

        public Builder title(String title) {
            this.title = title;
            return this;
        }

        public Builder header(String... header) {
            this.header = header;
            return this;
        }

        public Builder table(String[][] table) {
            this.table = table;
            return this;
        }

        public Builder border(int border) {
            this.border = border;
            return this;
        }

        public Builder padLeftPx(int padLeftPx) {
            this.padLeftPx = padLeftPx;
            return this;
        }

        public Builder padRightPx(int padRightPx) {
            this.padRightPx = padRightPx;
            return this;
        }

        public Builder padTopPx(int padTopPx) {
            this.padTopPx = padTopPx;
            return this;
        }

        public Builder padBottomPx(int padBottomPx) {
            this.padBottomPx = padBottomPx;
            return this;
        }

        public Builder paddingPx(int paddingPx) {
            padLeftPx(paddingPx);
            padRightPx(paddingPx);
            padTopPx(paddingPx);
            padBottomPx(paddingPx);
            return this;
        }

        public Builder paddingPx(int left, int right, int top, int bottom) {
            padLeftPx(left);
            padRightPx(right);
            padTopPx(top);
            padBottomPx(bottom);
            return this;
        }

        public Builder colWidthsPercent(double... colWidthsPercent) {
            this.colWidthsPercent = colWidthsPercent;
            return this;
        }

        public Builder backgroundColor(String backgroundColor) {
            this.backgroundColor = backgroundColor;
            return this;
        }

        public Builder headerColor(String headerColor) {
            this.headerColor = headerColor;
            return this;
        }

        public RenderableComponentTable build() {
            return new RenderableComponentTable(this);
        }

    }



}
