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

package org.datavec.arrow;

import lombok.val;
import org.apache.commons.io.FileUtils;
import org.datavec.api.records.mapper.RecordMapper;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.arrow.recordreader.ArrowRecordReader;
import org.datavec.arrow.recordreader.ArrowRecordWriter;
import org.junit.Test;
import org.nd4j.linalg.primitives.Triple;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class RecordMapperTest {

    @Test
    public void testMultiWrite() throws Exception {
        val recordsPair = records();

        Path p = Files.createTempFile("arrowwritetest", ".arrow");
        FileUtils.write(p.toFile(),recordsPair.getFirst());
        p.toFile().deleteOnExit();

        int numReaders = 2;
        RecordReader[] readers = new RecordReader[numReaders];
        InputSplit[] splits = new InputSplit[numReaders];
        for(int i = 0; i < readers.length; i++) {
            FileSplit split = new FileSplit(p.toFile());
            ArrowRecordReader arrowRecordReader = new ArrowRecordReader();
            readers[i] = arrowRecordReader;
            splits[i] = split;
        }

        ArrowRecordWriter arrowRecordWriter = new ArrowRecordWriter(recordsPair.getMiddle());
        FileSplit split = new FileSplit(p.toFile());
        arrowRecordWriter.initialize(split,new NumberOfRecordsPartitioner());
        arrowRecordWriter.writeBatch(recordsPair.getRight());


        CSVRecordWriter csvRecordWriter = new CSVRecordWriter();
        Path p2 = Files.createTempFile("arrowwritetest", ".csv");
        FileUtils.write(p2.toFile(),recordsPair.getFirst());
        p.toFile().deleteOnExit();
        FileSplit outputCsv = new FileSplit(p2.toFile());

        RecordMapper mapper = RecordMapper.builder().batchSize(10).inputUrl(split)
                .outputUrl(outputCsv)
                .partitioner(new NumberOfRecordsPartitioner()).readersToConcat(readers)
                .splitPerReader(splits)
                .recordWriter(csvRecordWriter)
                .build();
        mapper.copy();


    }


    @Test
    public void testCopyFromArrowToCsv() throws Exception {
        val recordsPair = records();

        Path p = Files.createTempFile("arrowwritetest", ".arrow");
        FileUtils.write(p.toFile(),recordsPair.getFirst());
        p.toFile().deleteOnExit();

        ArrowRecordWriter arrowRecordWriter = new ArrowRecordWriter(recordsPair.getMiddle());
        FileSplit split = new FileSplit(p.toFile());
        arrowRecordWriter.initialize(split,new NumberOfRecordsPartitioner());
        arrowRecordWriter.writeBatch(recordsPair.getRight());


        ArrowRecordReader arrowRecordReader = new ArrowRecordReader();
        arrowRecordReader.initialize(split);


        CSVRecordWriter csvRecordWriter = new CSVRecordWriter();
        Path p2 = Files.createTempFile("arrowwritetest", ".csv");
        FileUtils.write(p2.toFile(),recordsPair.getFirst());
        p.toFile().deleteOnExit();
        FileSplit outputCsv = new FileSplit(p2.toFile());

        RecordMapper mapper = RecordMapper.builder().batchSize(10).inputUrl(split)
                .outputUrl(outputCsv)
                .partitioner(new NumberOfRecordsPartitioner())
                .recordReader(arrowRecordReader).recordWriter(csvRecordWriter)
                .build();
        mapper.copy();

        CSVRecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(outputCsv);


        List<List<Writable>> loadedCSvRecords = recordReader.next(10);
        assertEquals(10,loadedCSvRecords.size());
    }


    @Test
    public void testCopyFromCsvToArrow() throws Exception {
        val recordsPair = records();

        Path p = Files.createTempFile("csvwritetest", ".csv");
        FileUtils.write(p.toFile(),recordsPair.getFirst());
        p.toFile().deleteOnExit();


        CSVRecordReader recordReader = new CSVRecordReader();
        FileSplit fileSplit = new FileSplit(p.toFile());

        ArrowRecordWriter arrowRecordWriter = new ArrowRecordWriter(recordsPair.getMiddle());
        File outputFile = Files.createTempFile("outputarrow","arrow").toFile();
        FileSplit outputFileSplit = new FileSplit(outputFile);
        RecordMapper mapper = RecordMapper.builder().batchSize(10).inputUrl(fileSplit)
                .outputUrl(outputFileSplit).partitioner(new NumberOfRecordsPartitioner())
                .recordReader(recordReader).recordWriter(arrowRecordWriter)
                .build();
        mapper.copy();

        ArrowRecordReader arrowRecordReader = new ArrowRecordReader();
        arrowRecordReader.initialize(outputFileSplit);
        List<List<Writable>> next = arrowRecordReader.next(10);
        System.out.println(next);
        assertEquals(10,next.size());

    }

    private Triple<String,Schema,List<List<Writable>>> records() {
        List<List<Writable>> list = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        int numColumns = 3;
        for (int i = 0; i < 10; i++) {
            List<Writable> temp = new ArrayList<>();
            for (int j = 0; j < numColumns; j++) {
                int v = 100 * i + j;
                temp.add(new IntWritable(v));
                sb.append(v);
                if (j < 2)
                    sb.append(",");
                else if (i != 9)
                    sb.append("\n");
            }
            list.add(temp);
        }


        Schema.Builder schemaBuilder = new Schema.Builder();
        for(int i = 0; i < numColumns; i++) {
            schemaBuilder.addColumnInteger(String.valueOf(i));
        }

        return Triple.of(sb.toString(),schemaBuilder.build(),list);
    }


}
