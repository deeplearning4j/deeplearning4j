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

import javafx.application.Application;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import javafx.util.StringConverter;
import org.apache.arrow.vector.FieldVector;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.arrow.recordreader.ArrowRecordReader;
import org.datavec.arrow.recordreader.ArrowWritableRecordBatch;
import org.nd4j.common.primitives.Counter;
import org.nd4j.common.primitives.CounterMap;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.profiler.data.eventlogger.EventType;
import org.nd4j.linalg.profiler.data.eventlogger.ObjectAllocationType;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.time.LocalDate;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class UnifiedProfilerAggregateLogAnalyzer extends Application  {

    private Counter<EventType> eventTypes = new Counter<>();
    private Counter<ObjectAllocationType> objectAllocationTypes = new Counter<>();
    private CounterMap<DataType,EventType> eventTypesByDataType = new CounterMap<>();
    private static File inputFile;
    private Stage stage;
    private static ZoneId defaultZoneId = ZoneId.systemDefault();
    private static //EEE MMM dd HH:mm:ss zzz
    DateTimeFormatter formatter = DateTimeFormatter.ofPattern("EEE MMM dd HH:mm:ss zzz",Locale.ENGLISH);
    public UnifiedProfilerAggregateLogAnalyzer(File inputFile, Stage stage) {
        this.inputFile = inputFile;
        this.stage = stage;
    }

    public UnifiedProfilerAggregateLogAnalyzer() {
    }

    public void analyze() throws Exception {
        Map<String,WorkspaceSeries> workspaceMemory = new ConcurrentHashMap<>();
        RuntimeSeries runtimeSeries = new RuntimeSeries();
        NumberAxis timeAxis = new NumberAxis();
        timeAxis.setLabel("Time (in ms)");
        timeAxis.setTickLabelFormatter(new StringConverter<Number>() {
            @Override
            public String toString(Number object) {
                return new Date(object.longValue()).toString();
            }

            @Override
            public Number fromString(String string) {
                LocalDate date = LocalDate.parse(string, DateTimeFormatter.ISO_DATE);
                Date date2 = Date.from(date.atStartOfDay(defaultZoneId).toInstant());
                return date2.getTime();

            }
        });

        //get the workspace names
        Schema schema = new Schema.Builder()
                .addColumnLong("eventTimeMs")
                .addColumnString("associatedWorkspace")
                .addColumnLong("workspaceSpilledBytes")
                .addColumnLong("workspacePinnedBytes")
                .addColumnLong("workspaceExternalBytes")
                .addColumnLong("workspaceAllocatedMemory")
                .addColumnLong("runtimeMaxMemory")
                .addColumnLong("runtimeFreeMemory")
                .addColumnLong("javacppMaxBytes")
                .addColumnLong("javacppMaxPhysicalBytes")
                .addColumnLong("javacppAvailablePhysicalBytes")
                .addColumnLong("javacppTotalBytes")
                .addColumnLong("javacppPointerCount")
                .build();


        ArrowRecordReader arrowRecordReader = new ArrowRecordReader();
        arrowRecordReader.initialize(new FileSplit(new File("arrow-output")));

        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .removeColumns(schema.getColumnNames().stream()
                        .filter(input -> input.equals("associatedWorkspace"))
                        .collect(Collectors.toList()))
                .build();

        TransformProcessRecordReader transformProcessRecordReader = new TransformProcessRecordReader(arrowRecordReader,transformProcess);
        Set<String> columns = new HashSet<>();
        while(transformProcessRecordReader.hasNext()) {
            List<Writable> next = transformProcessRecordReader.next();
            columns.addAll(next.stream().map(input -> input.toString()).collect(Collectors.toList()));
        }

        for(String column : columns) {
            WorkspaceSeries workspaceSeries = new WorkspaceSeries();
            workspaceMemory.put(column,workspaceSeries);
        }


        arrowRecordReader = new ArrowRecordReader();
        arrowRecordReader.initialize(new FileSplit(new File("arrow-output")));
        while(arrowRecordReader.hasNext()) {
            arrowRecordReader.next();
            ArrowWritableRecordBatch currentBatch = arrowRecordReader.getCurrentBatch();
            List<FieldVector> list = currentBatch.getList();
            runtimeSeries.record(list);
        }


        render(runtimeSeries.getRuntimeMaxMemory(),timeAxis,"Java heap max memory",new File("runtime-max-memory.png"));
        render(runtimeSeries.getJavacppAvailablePhysicalBytes(),timeAxis,"Javacpp available physical bytes",new File("javacpp-max-physical-bytes.png"));
        render(runtimeSeries.getJavacppMaxBytes(),timeAxis,"Javacpp maxphysical bytes",new File("javacpp-max-bytes.png"));
        render(runtimeSeries.getJavacppAvailablePhysicalBytes(),timeAxis,"Javacpp available physical bytes",new File("javacpp-available-physical-bytes.png"));
        render(runtimeSeries.getRuntimeFreeMemory(),timeAxis,"Java heap free memory",new File("runtime-free-memory.png"));

    }




    private void render(XYChart.Series<Number,Number> series,NumberAxis timeChart,String label,File outputFile) throws IOException {
        NumberAxis allocatedChart = new NumberAxis();
        allocatedChart.setLabel(label);
        LineChart<Number,Number> allocatedChart2 = new LineChart<>(allocatedChart,timeChart);
        allocatedChart2.getData().add(series);
        saveImage(allocatedChart2,outputFile);

    }

    private void saveImage(LineChart lineChart,File file) throws IOException {
        //https://stackoverflow.com/questions/29721289/how-to-generate-chart-image-using-javafx-chart-api-for-export-without-displying
        //save image as above
        VBox vbox = new VBox(lineChart);

        Scene scene = new Scene(vbox, 400, 200);

        stage.setScene(scene);
        stage.setHeight(300);
        stage.setWidth(1200);
        WritableImage image = scene.snapshot(null);
        ImageIO.write(SwingFXUtils.fromFXImage(image, null),
                "png", file);

    }

    public static void main(String...args) throws Exception {
        UnifiedProfilerAggregateLogAnalyzer.inputFile = new File(args[0]);
        Application.launch(args);
    }

    @Override
    public void start(Stage stage) throws Exception {
        UnifiedProfilerAggregateLogAnalyzer unifiedProfilerLogAnalyzer = new UnifiedProfilerAggregateLogAnalyzer(inputFile,stage);
        unifiedProfilerLogAnalyzer.analyze();
    }
}
