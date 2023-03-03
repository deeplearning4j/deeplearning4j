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
    import javafx.scene.image.WritableImage;
    import javafx.scene.layout.VBox;
    import javafx.stage.Stage;
    import org.datavec.api.split.FileSplit;
    import org.datavec.api.transform.analysis.DataAnalysis;
    import org.datavec.api.transform.schema.Schema;
    import org.datavec.api.transform.ui.HtmlAnalysis;
    import org.datavec.arrow.recordreader.ArrowRecordReader;
    import org.datavec.local.transforms.AnalyzeLocal;
    import org.nd4j.common.primitives.Counter;
    import org.nd4j.common.primitives.CounterMap;
    import org.nd4j.linalg.api.buffer.DataType;
    import org.nd4j.linalg.profiler.data.eventlogger.EventType;
    import org.nd4j.linalg.profiler.data.eventlogger.ObjectAllocationType;

    import javax.imageio.ImageIO;
    import java.io.File;
    import java.io.IOException;
    import java.util.Arrays;
    import java.util.Map;
    import java.util.concurrent.ConcurrentHashMap;
    import java.util.stream.Collectors;

    /**
     *
     */
    public class UnifiedProfilerLogAnalyzer extends Application  {

        private Counter<EventType> eventTypes = new Counter<>();
        private Counter<ObjectAllocationType> objectAllocationTypes = new Counter<>();
        private CounterMap<DataType,EventType> eventTypesByDataType = new CounterMap<>();
        private static File inputFile;
        private Stage stage;

        public UnifiedProfilerLogAnalyzer(File inputFile,Stage stage) {
            this.inputFile = inputFile;
            this.stage = stage;
        }

        public UnifiedProfilerLogAnalyzer() {
        }

        public void analyze() throws Exception {
            Map<String,WorkspaceSeries> workspaceMemory = new ConcurrentHashMap<>();
            RuntimeSeries runtimeSeries = new RuntimeSeries();
            NumberAxis timeAxis = new NumberAxis();

            //get the workspace names
            Schema schema = new Schema.Builder()
                    .addColumnLong("eventTimeMs")
                    .addColumnCategorical("eventType","ALLOCATION","DEALLOCATION")
                    .addColumnCategorical("objectAllocationType","OP_CONTEXT","DATA_BUFFER","WORKSPACE")
                    .addColumnString("associatedWorkspace")
                    .addColumnString("associatedThreadName")
                    .addColumnCategorical("datatype", Arrays.stream(DataType.values()).map(input -> input.name()).collect(Collectors.toList()).toArray(new String[0]))
                    .addColumnLong("memInBytes")
                    .addColumnBoolean("isAttached")
                    .addColumnBoolean("isConstant")
                    .addColumnLong("objectId")
                    .addColumnLong("workspaceAllocatedMemory")
                    .addColumnLong("workspaceExternalBytes")
                    .addColumnLong("workspacePinnedBytes")
                    .addColumnLong("workspaceSpilledBytes")
                    .addColumnLong("runtimeFreeMemory")
                    .addColumnLong("javacppAvailablePhysicalBytes")
                    .addColumnLong("javacppMaxPhysicalBytes")
                    .addColumnLong("javacppMaxBytes")
                    .addColumnLong("runtimeMaxMemory")
                    .build();

            ArrowRecordReader arrowRecordReader = new ArrowRecordReader();
            arrowRecordReader.initialize(new FileSplit(new File("arrow-output")));
            DataAnalysis analyze = AnalyzeLocal.analyze(schema, arrowRecordReader);
            HtmlAnalysis.createHtmlAnalysisFile(analyze,new File("analysis.html"));


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
            UnifiedProfilerLogAnalyzer.inputFile = new File(args[0]);
            Application.launch(args);
        }

        @Override
        public void start(Stage stage) throws Exception {
            UnifiedProfilerLogAnalyzer unifiedProfilerLogAnalyzer = new UnifiedProfilerLogAnalyzer(inputFile,stage);
            unifiedProfilerLogAnalyzer.analyze();
        }
    }
