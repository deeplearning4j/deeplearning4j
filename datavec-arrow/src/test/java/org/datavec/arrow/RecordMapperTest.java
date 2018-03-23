package org.datavec.arrow;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.mapper.RecordMapper;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.arrow.recordreader.ArrowRecordReader;
import org.datavec.arrow.recordreader.ArrowRecordWriter;
import org.junit.Test;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class RecordMapperTest {

    @Test
    public void testCopyFromCsvToArrow() throws Exception {
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


        Path p = Files.createTempFile("csvwritetest", ".csv");
        FileUtils.write(p.toFile(),sb.toString());
        p.toFile().deleteOnExit();

        Schema.Builder schemaBuilder = new Schema.Builder();
        for(int i = 0; i < numColumns; i++) {
            schemaBuilder.addColumnInteger(String.valueOf(i));
        }

        CSVRecordReader recordReader = new CSVRecordReader();
        FileSplit fileSplit = new FileSplit(p.toFile());

        ArrowRecordWriter arrowRecordWriter = new ArrowRecordWriter(schemaBuilder.build());
        File outputFile = Files.createTempFile("outputarrow","arrow").toFile();
        FileSplit outputFileSplit = new FileSplit(outputFile);
        RecordMapper mapper = RecordMapper.builder().batchSize(5).inputUrl(fileSplit)
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

}
