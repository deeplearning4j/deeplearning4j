
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
import javafx.scene.chart.XYChart.Series;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.UInt8Vector;
import org.nd4j.linalg.profiler.data.eventlogger.LogEvent;

import java.util.List;
import java.util.Objects;

/**
 * Aggregates metrics for global runtime metrics for rendering.
 * @author Adam Gibson
 */
public class RuntimeSeries {
    private Series
            runtimeFreeMemory,
            runtimeMaxMemory,
            javacppAvailablePhysicalBytes,
            javacppMaxBytes,
            javacppMaxPhysicalBytes,
            javacppTotalBytes,
            javacppPointerCount;


    public RuntimeSeries() {
        runtimeFreeMemory = new Series();
        runtimeMaxMemory = new Series();
        javacppAvailablePhysicalBytes = new Series();
        javacppMaxBytes = new Series();
        javacppMaxPhysicalBytes = new Series();
        javacppTotalBytes = new Series();
        javacppPointerCount = new Series();
    }


    /**
     * Use arrow vectors to record metrics.
     * @param inputVectors
     */
    public synchronized void record(List<FieldVector> inputVectors) {
        UInt8Vector eventTimeMsVec = (UInt8Vector) inputVectors.get(0);
        UInt8Vector runtimeMaxMemoryVec = (UInt8Vector) inputVectors.get(6);
        UInt8Vector runtimeFreeMemoryVec = (UInt8Vector) inputVectors.get(7);
        UInt8Vector javacppMaxBytesVec = (UInt8Vector) inputVectors.get(8);
        UInt8Vector javacppMaxPhysicalBytesVec = (UInt8Vector) inputVectors.get(9);
        UInt8Vector javacppAvailablePhysicalBytesVec = (UInt8Vector) inputVectors.get(10);
        UInt8Vector javacppTotalBytesVec = (UInt8Vector) inputVectors.get(11);
        UInt8Vector javacppPointerCountVec = (UInt8Vector) inputVectors.get(12);
        for(int i  = 0; i < inputVectors.get(0).getValueCount(); i++) {
            long eventTimeMs = eventTimeMsVec.get(i);
            long runtimeMaxMemoryVal = runtimeMaxMemoryVec.get(i);
            long runtimeFreeMemoryVal = runtimeFreeMemoryVec.get(i);
            long javacppMaxBytesVal = javacppMaxBytesVec.get(i);
            long javacppMaxPhysicalBytesVal = javacppMaxPhysicalBytesVec.get(i);
            long javacppAvailablePhysicalBytesVal = javacppAvailablePhysicalBytesVec.get(i);
            long javacppPointerCountVal = javacppPointerCountVec.get(i);
            long javacppTotalBytesVal = javacppTotalBytesVec.get(i);
            runtimeFreeMemory.getData().add(new XYChart.Data<>(runtimeFreeMemoryVal,eventTimeMs));
            runtimeMaxMemory.getData().add(new XYChart.Data<>(runtimeMaxMemoryVal,eventTimeMs));
            javacppAvailablePhysicalBytes.getData().add(new XYChart.Data<>(javacppAvailablePhysicalBytesVal,eventTimeMs));
            javacppMaxBytes.getData().add(new XYChart.Data<>(javacppMaxBytesVal,eventTimeMs));
            javacppMaxPhysicalBytes.getData().add(new XYChart.Data<>(javacppMaxPhysicalBytesVal,eventTimeMs));
            javacppTotalBytes.getData().add(new XYChart.Data<>(javacppTotalBytesVal,eventTimeMs));
            javacppPointerCount.getData().add(new XYChart.Data<>(javacppPointerCountVal,eventTimeMs));

        }
    }

    public synchronized void record(LogEvent logEvent) {
        runtimeFreeMemory.getData().add(new XYChart.Data<>(logEvent.getRunTimeMemory().getRuntimeFreeMemory(),logEvent.getEventTimeMs()));
        runtimeMaxMemory.getData().add(new XYChart.Data<>(logEvent.getRunTimeMemory().getRuntimeMaxMemory(),logEvent.getEventTimeMs()));
        javacppAvailablePhysicalBytes.getData().add(new XYChart.Data<>(logEvent.getRunTimeMemory().getJavacppAvailablePhysicalBytes(),logEvent.getEventTimeMs()));
        javacppMaxBytes.getData().add(new XYChart.Data<>(logEvent.getRunTimeMemory().getJavacppMaxBytes(),logEvent.getEventTimeMs()));
        javacppMaxPhysicalBytes.getData().add(new XYChart.Data<>(logEvent.getRunTimeMemory().getJavacppAvailablePhysicalBytes(),logEvent.getEventTimeMs()));
        javacppTotalBytes.getData().add(new XYChart.Data<>(logEvent.getRunTimeMemory().getJavacppTotalBytes(),logEvent.getEventTimeMs()));
        javacppPointerCount.getData().add(new XYChart.Data<>(logEvent.getRunTimeMemory().getJavacppPointerCount(),logEvent.getEventTimeMs()));

    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        RuntimeSeries that = (RuntimeSeries) o;
        return Objects.equals(runtimeFreeMemory, that.runtimeFreeMemory) && Objects.equals(runtimeMaxMemory, that.runtimeMaxMemory) && Objects.equals(javacppAvailablePhysicalBytes, that.javacppAvailablePhysicalBytes) && Objects.equals(javacppMaxBytes, that.javacppMaxBytes) && Objects.equals(javacppMaxPhysicalBytes, that.javacppMaxPhysicalBytes);
    }

    @Override
    public int hashCode() {
        return Objects.hash(runtimeFreeMemory, runtimeMaxMemory, javacppAvailablePhysicalBytes, javacppMaxBytes, javacppMaxPhysicalBytes);
    }

    public XYChart.Series getRuntimeFreeMemory() {
        return runtimeFreeMemory;
    }

    public void setRuntimeFreeMemory(XYChart.Series runtimeFreeMemory) {
        this.runtimeFreeMemory = runtimeFreeMemory;
    }

    public XYChart.Series getRuntimeMaxMemory() {
        return runtimeMaxMemory;
    }

    public void setRuntimeMaxMemory(XYChart.Series runtimeMaxMemory) {
        this.runtimeMaxMemory = runtimeMaxMemory;
    }

    public XYChart.Series getJavacppAvailablePhysicalBytes() {
        return javacppAvailablePhysicalBytes;
    }

    public void setJavacppAvailablePhysicalBytes(XYChart.Series javacppAvailablePhysicalBytes) {
        this.javacppAvailablePhysicalBytes = javacppAvailablePhysicalBytes;
    }

    public XYChart.Series getJavacppMaxBytes() {
        return javacppMaxBytes;
    }

    public void setJavacppMaxBytes(XYChart.Series javacppMaxBytes) {
        this.javacppMaxBytes = javacppMaxBytes;
    }

    public XYChart.Series getJavacppMaxPhysicalBytes() {
        return javacppMaxPhysicalBytes;
    }

    public void setJavacppMaxPhysicalBytes(XYChart.Series javacppMaxPhysicalBytes) {
        this.javacppMaxPhysicalBytes = javacppMaxPhysicalBytes;
    }
}
