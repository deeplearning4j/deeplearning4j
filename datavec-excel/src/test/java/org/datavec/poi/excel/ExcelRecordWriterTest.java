package org.datavec.poi.excel;

import lombok.val;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.junit.Test;
import org.nd4j.linalg.primitives.Triple;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class ExcelRecordWriterTest {

    @Test
    public void testWriter() throws Exception  {
        ExcelRecordWriter excelRecordWriter = new ExcelRecordWriter();
        val records = records();
        File tmpDir = Files.createTempDirectory("testexcel").toFile();
        File outputFile = new File(tmpDir,"testexcel.xlsx");
        outputFile.deleteOnExit();
        FileSplit fileSplit = new FileSplit(outputFile);
        excelRecordWriter.initialize(fileSplit,new NumberOfRecordsPartitioner());
        excelRecordWriter.writeBatch(records.getRight());
        excelRecordWriter.close();
        File parentFile = outputFile.getParentFile();
        assertEquals(1,parentFile.list().length);

        ExcelRecordReader excelRecordReader = new ExcelRecordReader();
        excelRecordReader.initialize(fileSplit);
        List<List<Writable>> next = excelRecordReader.next(10);
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
