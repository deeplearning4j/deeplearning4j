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

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.ui.api.Utils;
import org.deeplearning4j.ui.components.chart.style.StyleChart;
import org.nd4j.shade.jackson.annotation.JsonInclude;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

/**
 * A timeline/swimlane chart with zoom/scroll functionality.
 *
 * Time is represented here with long values - i.e., millisecond precision, epoch format
 *
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ChartTimeline extends Chart {
    public static final String COMPONENT_TYPE = "ChartTimeline";

    private List<String> laneNames = new ArrayList<>();
    private List<List<TimelineEntry>> laneData = new ArrayList<>();

    public ChartTimeline() {
        super(COMPONENT_TYPE);
        //no-arg constructor for Jackson
    }

    private ChartTimeline(Builder builder) {
        super(COMPONENT_TYPE, builder);
        this.laneNames = builder.laneNames;
        this.laneData = builder.laneData;
    }



    public static class Builder extends Chart.Builder<Builder> {
        private List<String> laneNames = new ArrayList<>();
        private List<List<TimelineEntry>> laneData = new ArrayList<>();


        public Builder(String title, StyleChart style) {
            super(title, style);
        }


        public Builder addLane(String name, List<TimelineEntry> data) {
            laneNames.add(name);
            laneData.add(data);
            return this;
        }

        public ChartTimeline build() {
            return new ChartTimeline(this);
        }
    }


    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static class TimelineEntry {
        private String entryLabel;
        private long startTimeMs;
        private long endTimeMs;
        private String color;

        public TimelineEntry(String entryLabel, long startTimeMs, long endTimeMs) {
            this.entryLabel = entryLabel;
            this.startTimeMs = startTimeMs;
            this.endTimeMs = endTimeMs;
        }

        public TimelineEntry(String entryLabel, long startTimeMs, long endTimeMs, Color color) {
            this.entryLabel = entryLabel;
            this.startTimeMs = startTimeMs;
            this.endTimeMs = endTimeMs;
            this.color = Utils.colorToHex(color);
        }

    }
}
