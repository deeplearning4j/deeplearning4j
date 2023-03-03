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

import org.datavec.api.records.mapper.RecordMapper;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.transform.schema.Schema;
import org.datavec.arrow.recordreader.ArrowRecordWriter;
import org.nd4j.linalg.api.buffer.DataType;

import java.io.*;
import java.util.Arrays;
import java.util.Scanner;
import java.util.stream.Collectors;

/**
 * Converts csv logs from {@link org.nd4j.linalg.profiler.data.eventlogger.EventLogger}
 * when {@link org.nd4j.linalg.profiler.data.eventlogger.EventLogger#aggregateMode}
 * is false to arrow format.
 *
 * @author Adam Gibson
 */
public class UnifiedProfilerArrowConverter {

    public final static String SPLIT_CSV_DIRECTORY = "split-csv";

    public static void main(String... args) throws Exception {
        Schema schema = new Schema.Builder()
                .addColumnLong("eventTimeMs")
                .addColumnCategorical("eventType", "ALLOCATION", "DEALLOCATION")
                .addColumnCategorical("objectAllocationType", "OP_CONTEXT", "DATA_BUFFER", "WORKSPACE")
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
                .addColumnLong("javacppPointerCount")
                .addColumnLong("javacppTotalBytes")
                .addColumnLong("runtimeMaxMemory")
                .build();


        File logFile = new File("/home/agibsonccc/Documents/GitHub/servicenow-reproducer/event-log.csv");

        splitLogFile(logFile,10000);
        CSVRecordReader recordReader = new CSVRecordReader();

        File[] inputFiles = new File(SPLIT_CSV_DIRECTORY).listFiles();
        for(File inputFile : inputFiles) {
            FileSplit inputCSv = new FileSplit(inputFile);
            ArrowRecordWriter arrowRecordWriter = new ArrowRecordWriter(schema);
            File outputFile = new File("arrow-output/",inputFile.getName() + ".arrow");
            outputFile.createNewFile();
            FileSplit outputFileSplit = new FileSplit(outputFile);

            RecordMapper recordMapper = RecordMapper.builder()
                    .recordReader(recordReader)
                    .recordWriter(arrowRecordWriter)
                    .inputUrl(inputCSv)
                    .batchSize(10000)
                    .partitioner(new NumberOfRecordsPartitioner())
                    .outputUrl(outputFileSplit)
                    .callInitRecordReader(true)
                    .callInitRecordWriter(true)
                    .build();
            recordMapper.copy();
        }


    }

    private static void splitLogFile(File logFile, double numLinesPerFile) throws IOException {
        Scanner scanner = new Scanner(logFile);
        int count = 0;
        while (scanner.hasNextLine()) {
            scanner.nextLine();
            count++;
        }
        System.out.println("Lines in the file: " + count);     // Displays no. of lines in the input file.

        double temp = (count / numLinesPerFile);
        int temp1 = (int) temp;
        int nof = 0;
        if (temp1 == temp) {
            nof = temp1;
        } else {
            nof = temp1 + 1;
        }
        System.out.println("No. of files to be generated :" + nof); // Displays no. of files to be generated.

        //---------------------------------------------------------------------------------------------------------

        // Actual splitting of file into smaller files

        FileInputStream fstream = new FileInputStream(logFile);
        DataInputStream in = new DataInputStream(fstream);

        BufferedReader br = new BufferedReader(new InputStreamReader(in));
        String strLine;
        File inputDirectory = new File(SPLIT_CSV_DIRECTORY);
        if(!inputDirectory.exists())
            inputDirectory.mkdirs();
        for (int j = 1; j <= nof; j++) {
            FileWriter fstream1 = new FileWriter(SPLIT_CSV_DIRECTORY + File.separator + "event-log-" + j + ".csv");     // Destination File Location
            BufferedWriter out = new BufferedWriter(fstream1);
            for (int i = 1; i <= numLinesPerFile; i++) {
                strLine = br.readLine();
                if (strLine != null) {
                    out.write(strLine);
                    if (i != numLinesPerFile) {
                        out.newLine();
                    }
                }
            }
            out.close();
        }

        in.close();

    }

}
