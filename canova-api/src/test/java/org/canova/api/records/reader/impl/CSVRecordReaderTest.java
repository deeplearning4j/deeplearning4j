package org.canova.api.records.reader.impl;

import org.apache.commons.io.FileUtils;
import org.canova.api.io.data.IntWritable;
import org.canova.api.io.data.Text;
import org.canova.api.records.writer.impl.CSVRecordWriter;
import org.canova.api.records.writer.impl.FileRecordWriter;
import org.canova.api.split.FileSplit;
import org.canova.api.split.StringSplit;
import org.canova.api.util.ClassPathResource;
import org.canova.api.writable.Writable;
import org.junit.Test;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.*;

public class CSVRecordReaderTest {
    @Test
    public void testNext() throws Exception {
        CSVRecordReader reader = new CSVRecordReader();
        reader.initialize(new StringSplit("1,1,8.0,,,,14.0,,,,15.0,,,,,,,,,,,,1"));
        while (reader.hasNext()) {
            Collection<Writable> vals = reader.next();
            List<Writable> arr = new ArrayList<>(vals);

            assertEquals("Entry count", 23, vals.size());
            Text lastEntry = (Text)arr.get(arr.size()-1);
            assertEquals("Last entry garbage", 1, lastEntry.getLength());
        }
    }

    @Test
    public void testEmptyEntries() throws Exception {
        CSVRecordReader reader = new CSVRecordReader();
        reader.initialize(new StringSplit("1,1,8.0,,,,14.0,,,,15.0,,,,,,,,,,,,"));
        while (reader.hasNext()) {
            Collection<Writable> vals = reader.next();
            assertEquals("Entry count", 23, vals.size());
        }
    }

    @Test
    public void testReset() throws Exception {
        CSVRecordReader rr = new CSVRecordReader(0,",");
        rr.initialize(new FileSplit(new ClassPathResource("iris.dat").getFile()));

        int nResets = 5;
        for( int i=0; i < nResets; i++ ){

            int lineCount = 0;
            while(rr.hasNext()){
                Collection<Writable> line = rr.next();
                assertEquals(5, line.size());
                lineCount++;
            }
            assertFalse(rr.hasNext());
            assertEquals(150, lineCount);
            rr.reset();
        }
    }

    @Test
    public void testResetWithSkipLines() throws Exception {
        CSVRecordReader rr = new CSVRecordReader(10,",");
        rr.initialize(new FileSplit(new ClassPathResource("iris.dat").getFile()));
        int lineCount = 0;
        while(rr.hasNext()) {
            rr.next();
            ++lineCount;
        }
        assertEquals(140, lineCount);
        rr.reset();
        lineCount = 0;
        while(rr.hasNext()) {
            rr.next();
            ++lineCount;
        }
        assertEquals(140, lineCount);
    }

    @Test
    public void testWrite() throws Exception {

        List<Collection<Writable>> list = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        for( int i=0; i<10; i++ ){
            List<Writable> temp = new ArrayList<>();
            for( int j=0; j<3; j++ ){
                int v = 100*i+j;
                temp.add(new IntWritable(v));
                sb.append(v);
                if(j < 2) sb.append(",");
                else if(i != 9) sb.append("\n");
            }
            list.add(temp);
        }

        String expected = sb.toString();

        Path p = Files.createTempFile("csvwritetest","csv");
        p.toFile().deleteOnExit();

        FileRecordWriter writer = new CSVRecordWriter(p.toFile());
        for( Collection<Writable> c : list ){
            writer.write(c);
        }
        writer.close();

        //Read file back in; compare
        String fileContents = FileUtils.readFileToString(p.toFile(),FileRecordWriter.DEFAULT_CHARSET.name());

        System.out.println(expected);
        System.out.println("----------");
        System.out.println(fileContents);

        assertEquals(expected,fileContents);
    }

    @Test
    public void testTabsAsSplit1() throws Exception {

        CSVRecordReader reader = new CSVRecordReader(0,"\t");
        reader.initialize(new FileSplit(new ClassPathResource("/tabbed.txt").getFile()));
        while (reader.hasNext()) {
            List<Writable> list = new ArrayList<>(reader.next());

            assertEquals(2, list.size());
        }
    }
}
