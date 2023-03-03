/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.profiler.unifiedprofiler.analysis;

import javafx.scene.chart.XYChart;
import org.nd4j.linalg.profiler.data.eventlogger.LogEvent;

import java.util.Objects;

/**
 * Aggregates workspace metrics
 * from {@link org.nd4j.linalg.profiler.data.eventlogger.EventLogger}
 * when {@link org.nd4j.linalg.profiler.data.eventlogger.EventLogger#aggregateMode}
 * is false.
 *
 * @author Adam Gibson
 */

public class WorkspaceSeries {
    private XYChart.Series spilled,external,allocated,pinned;

    private long eventTimeMs;
    public WorkspaceSeries() {
        spilled = new XYChart.Series();
        external = new XYChart.Series();
        allocated = new XYChart.Series();
        pinned = new XYChart.Series();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        WorkspaceSeries that = (WorkspaceSeries) o;
        return eventTimeMs == that.eventTimeMs && Objects.equals(spilled, that.spilled) && Objects.equals(external, that.external) && Objects.equals(allocated, that.allocated) && Objects.equals(pinned, that.pinned);
    }

    @Override
    public int hashCode() {
        return Objects.hash(spilled, external, allocated, pinned, eventTimeMs);
    }

    public long getEventTimeMs() {
        return eventTimeMs;
    }

    public void setEventTimeMs(long eventTimeMs) {
        this.eventTimeMs = eventTimeMs;
    }

    public XYChart.Series getSpilled() {
        return spilled;
    }

    public void setSpilled(XYChart.Series spilled) {
        this.spilled = spilled;
    }

    public XYChart.Series getExternal() {
        return external;
    }

    public void setExternal(XYChart.Series external) {
        this.external = external;
    }

    public XYChart.Series getAllocated() {
        return allocated;
    }

    public void setAllocated(XYChart.Series allocated) {
        this.allocated = allocated;
    }

    public XYChart.Series getPinned() {
        return pinned;
    }

    public void setPinned(XYChart.Series pinned) {
        this.pinned = pinned;
    }

    public synchronized void record(LogEvent logEvent) {
        spilled.getData().add(new XYChart.Data(logEvent.getEventTimeMs(),logEvent.getWorkspaceInfo().getSpilledBytes(),logEvent.getEventTimeMs()));
        external.getData().add(new XYChart.Data(logEvent.getEventTimeMs(),logEvent.getWorkspaceInfo().getExternalBytes(),logEvent.getEventTimeMs()));
        allocated.getData().add(new XYChart.Data(logEvent.getEventTimeMs(),logEvent.getWorkspaceInfo().getAllocatedMemory(),logEvent.getEventTimeMs()));
        pinned.getData().add(new XYChart.Data(logEvent.getEventTimeMs(),logEvent.getWorkspaceInfo().getPinnedBytes(),logEvent.getEventTimeMs()));
    }
    

}
