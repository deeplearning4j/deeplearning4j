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

package org.datavec.api.transform.ui.components;

import lombok.Data;
import lombok.EqualsAndHashCode;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Alex on 25/03/2016.
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class RenderableComponentHistogram extends RenderableComponent {
    public static final String COMPONENT_TYPE = "histogram";

    private String title;
    private List<Double> lowerBounds = new ArrayList<>();
    private List<Double> upperBounds = new ArrayList<>();
    private List<Double> yValues = new ArrayList<>();
    private int marginTop;
    private int marginBottom;
    private int marginLeft;
    private int marginRight;

    public RenderableComponentHistogram(Builder builder) {
        super(COMPONENT_TYPE);
        this.title = builder.title;
        this.lowerBounds = builder.lowerBounds;
        this.upperBounds = builder.upperBounds;
        this.yValues = builder.yValues;
        this.marginTop = builder.marginTop;
        this.marginBottom = builder.marginBottom;
        this.marginLeft = builder.marginLeft;
        this.marginRight = builder.marginRight;
    }


    public static class Builder {

        private String title;
        private List<Double> lowerBounds = new ArrayList<>();
        private List<Double> upperBounds = new ArrayList<>();
        private List<Double> yValues = new ArrayList<>();
        private int marginTop = 60;
        private int marginBottom = 60;
        private int marginLeft = 60;
        private int marginRight = 20;

        public Builder title(String title) {
            this.title = title;
            return this;
        }

        public Builder addBin(double lower, double upper, double yValue) {
            lowerBounds.add(lower);
            upperBounds.add(upper);
            yValues.add(yValue);
            return this;
        }

        public Builder margins(int top, int bottom, int left, int right) {
            this.marginTop = top;
            this.marginBottom = bottom;
            this.marginLeft = left;
            this.marginRight = right;
            return this;
        }

        public RenderableComponentHistogram build() {
            return new RenderableComponentHistogram(this);
        }
    }
}
