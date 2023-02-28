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

import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import org.nd4j.common.primitives.Counter;
import org.nd4j.common.primitives.CounterMap;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.profiler.data.eventlogger.EventType;
import org.nd4j.linalg.profiler.data.eventlogger.LogEvent;
import org.nd4j.linalg.profiler.data.eventlogger.ObjectAllocationType;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class UnifiedProfilerLogAnalyzer {

    private Counter<EventType> eventTypes = new Counter<>();
    private Counter<ObjectAllocationType> objectAllocationTypes = new Counter<>();
    private long endMemory;
    private CounterMap<DataType,EventType> eventTypesByDataType = new CounterMap<>();
    private File inputFile;

    public UnifiedProfilerLogAnalyzer(File inputFile) {
        this.inputFile = inputFile;
    }


    public void analyze() throws IOException {
        NumberAxis memoryChart = new NumberAxis();
        memoryChart.setLabel("Memory in MB");
        XYChart.Series memoryData = new XYChart.Series();

        NumberAxis timeChart = new NumberAxis();
        XYChart.Series timeData = new XYChart.Series();
        timeChart.setLabel("Time chart");

        Files.readAllLines(inputFile.toPath()).stream()
                .map(LogEvent::eventFromLine).forEach(logEvent -> {
                    memoryData.getData().add(logEvent.getBytes());
                    timeData.getData().add(logEvent.getEventTimeMs());
                    eventTypes.incrementCount(logEvent.getEventType(),1.0);
                    objectAllocationTypes.incrementCount(logEvent.getObjectAllocationType(),1.0);
                    eventTypesByDataType.incrementCount(logEvent.getDataType(),logEvent.getEventType(),1.0);
                });

        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("Event types: \n");
        stringBuilder.append("Allocation counts: " + eventTypes.getCount(EventType.ALLOCATION) + "\n");
        stringBuilder.append("Deallocation counts: " + eventTypes.getCount(EventType.DEALLOCATION) + "\n");

        stringBuilder.append("Object allocation counts:\n");
        stringBuilder.append("Workspaces: " + objectAllocationTypes.getCount(ObjectAllocationType.WORKSPACE) + "\n");
        stringBuilder.append("Op contexts: " + objectAllocationTypes.getCount(ObjectAllocationType.OP_CONTEXT) + "\n");
        stringBuilder.append("Data buffers: " + objectAllocationTypes.getCount(ObjectAllocationType.DATA_BUFFER) + "\n");
        stringBuilder.append("Event type by data type:\n");
        for(DataType dataType : DataType.values()) {
            for(EventType eventType : EventType.values()) {
                stringBuilder.append("Data type: " + dataType + " Event type: " + eventType + " " + eventTypesByDataType.getCount(dataType,eventType) + " \n");
            }
        }



        System.out.println(stringBuilder);
    }

    public static void main(String...args) throws Exception {
        if(args.length < 1) {
            return;
        }

        File file = new File(args[0]);
        UnifiedProfilerLogAnalyzer unifiedProfilerLogAnalyzer = new UnifiedProfilerLogAnalyzer(file);
        unifiedProfilerLogAnalyzer.analyze();
    }

}
