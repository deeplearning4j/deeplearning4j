/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.records.reader.impl;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRegexRecordReader;
import org.datavec.api.records.writer.impl.FileRecordWriter;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputStreamInputSplit;
import org.datavec.api.split.StringSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

import static org.junit.Assert.*;

public class CSVRecordReaderTest {
    @Test
    public void testNext() throws Exception {
        CSVRecordReader reader = new CSVRecordReader();
        reader.initialize(new StringSplit("1,1,8.0,,,,14.0,,,,15.0,,,,,,,,,,,,1"));
        while (reader.hasNext()) {
            List<Writable> vals = reader.next();
            List<Writable> arr = new ArrayList<>(vals);

            assertEquals("Entry count", 23, vals.size());
            Text lastEntry = (Text) arr.get(arr.size() - 1);
            assertEquals("Last entry garbage", 1, lastEntry.getLength());
        }
    }

    @Test
    public void testEmptyEntries() throws Exception {
        CSVRecordReader reader = new CSVRecordReader();
        reader.initialize(new StringSplit("1,1,8.0,,,,14.0,,,,15.0,,,,,,,,,,,,"));
        while (reader.hasNext()) {
            List<Writable> vals = reader.next();
            assertEquals("Entry count", 23, vals.size());
        }
    }

    @Test
    public void testReset() throws Exception {
        CSVRecordReader rr = new CSVRecordReader(0, ',');
        rr.initialize(new FileSplit(new ClassPathResource("iris.dat").getFile()));

        int nResets = 5;
        for (int i = 0; i < nResets; i++) {

            int lineCount = 0;
            while (rr.hasNext()) {
                List<Writable> line = rr.next();
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
        CSVRecordReader rr = new CSVRecordReader(10, ',');
        rr.initialize(new FileSplit(new ClassPathResource("iris.dat").getFile()));
        int lineCount = 0;
        while (rr.hasNext()) {
            rr.next();
            ++lineCount;
        }
        assertEquals(140, lineCount);
        rr.reset();
        lineCount = 0;
        while (rr.hasNext()) {
            rr.next();
            ++lineCount;
        }
        assertEquals(140, lineCount);
    }

    @Test
    public void testWrite() throws Exception {
        List<List<Writable>> list = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 10; i++) {
            List<Writable> temp = new ArrayList<>();
            for (int j = 0; j < 3; j++) {
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

        String expected = sb.toString();

        Path p = Files.createTempFile("csvwritetest", "csv");
        p.toFile().deleteOnExit();

        FileRecordWriter writer = new CSVRecordWriter();
        FileSplit fileSplit = new FileSplit(p.toFile());
        writer.initialize(fileSplit,new NumberOfRecordsPartitioner());
        for (List<Writable> c : list) {
            writer.write(c);
        }
        writer.close();

        //Read file back in; compare
        String fileContents = FileUtils.readFileToString(p.toFile(), FileRecordWriter.DEFAULT_CHARSET.name());

        //        System.out.println(expected);
        //        System.out.println("----------");
        //        System.out.println(fileContents);

        assertEquals(expected, fileContents);
    }

    @Test
    public void testTabsAsSplit1() throws Exception {

        CSVRecordReader reader = new CSVRecordReader(0, '\t');
        reader.initialize(new FileSplit(new ClassPathResource("/tabbed.txt").getFile()));
        while (reader.hasNext()) {
            List<Writable> list = new ArrayList<>(reader.next());

            assertEquals(2, list.size());
        }
    }

    @Test
    public void testPipesAsSplit() throws Exception {

        CSVRecordReader reader = new CSVRecordReader(0, '|');
        reader.initialize(new FileSplit(new ClassPathResource("issue414.csv").getFile()));
        int lineidx = 0;
        List<Integer> sixthColumn = Arrays.asList(13, 95, 15, 25);
        while (reader.hasNext()) {
            List<Writable> list = new ArrayList<>(reader.next());

            assertEquals(10, list.size());
            assertEquals((long)sixthColumn.get(lineidx), list.get(5).toInt());
            lineidx++;
        }
    }


    @Test
    public void testWithQuotes() throws Exception {
        CSVRecordReader reader = new CSVRecordReader(0, ',', '\"');
        reader.initialize(new StringSplit("1,0,3,\"Braund, Mr. Owen Harris\",male,\"\"\"\""));
        while (reader.hasNext()) {
            List<Writable> vals = reader.next();
            assertEquals("Entry count", 6, vals.size());
            assertEquals("1", vals.get(0).toString());
            assertEquals("0", vals.get(1).toString());
            assertEquals("3", vals.get(2).toString());
            assertEquals("Braund, Mr. Owen Harris", vals.get(3).toString());
            assertEquals("male", vals.get(4).toString());
            assertEquals("\"", vals.get(5).toString());
        }
    }


    @Test
    public void testMeta() throws Exception {
        CSVRecordReader rr = new CSVRecordReader(0, ',');
        rr.initialize(new FileSplit(new ClassPathResource("iris.dat").getFile()));

        int lineCount = 0;
        List<RecordMetaData> metaList = new ArrayList<>();
        List<List<Writable>> writables = new ArrayList<>();
        while (rr.hasNext()) {
            Record r = rr.nextRecord();
            assertEquals(5, r.getRecord().size());
            lineCount++;
            RecordMetaData meta = r.getMetaData();
            //            System.out.println(r.getRecord() + "\t" + meta.getLocation() + "\t" + meta.getURI());

            metaList.add(meta);
            writables.add(r.getRecord());
        }
        assertFalse(rr.hasNext());
        assertEquals(150, lineCount);
        rr.reset();


        System.out.println("\n\n\n--------------------------------");
        List<Record> contents = rr.loadFromMetaData(metaList);
        assertEquals(150, contents.size());
        //        for(Record r : contents ){
        //            System.out.println(r);
        //        }

        List<RecordMetaData> meta2 = new ArrayList<>();
        meta2.add(metaList.get(100));
        meta2.add(metaList.get(90));
        meta2.add(metaList.get(80));
        meta2.add(metaList.get(70));
        meta2.add(metaList.get(60));

        List<Record> contents2 = rr.loadFromMetaData(meta2);
        assertEquals(writables.get(100), contents2.get(0).getRecord());
        assertEquals(writables.get(90), contents2.get(1).getRecord());
        assertEquals(writables.get(80), contents2.get(2).getRecord());
        assertEquals(writables.get(70), contents2.get(3).getRecord());
        assertEquals(writables.get(60), contents2.get(4).getRecord());
    }

    @Test
    public void testRegex() throws Exception {
        CSVRecordReader reader = new CSVRegexRecordReader(0, ",", null, new String[] {null, "(.+) (.+) (.+)"});
        reader.initialize(new StringSplit("normal,1.2.3.4 space separator"));
        while (reader.hasNext()) {
            List<Writable> vals = reader.next();
            assertEquals("Entry count", 4, vals.size());
            assertEquals("normal", vals.get(0).toString());
            assertEquals("1.2.3.4", vals.get(1).toString());
            assertEquals("space", vals.get(2).toString());
            assertEquals("separator", vals.get(3).toString());
        }
    }

    @Test(expected = NoSuchElementException.class)
    public void testCsvSkipAllLines() throws IOException, InterruptedException {
        final int numLines = 4;
        final List<Writable> lineList = Arrays.asList((Writable) new IntWritable(numLines - 1),
                        (Writable) new Text("one"), (Writable) new Text("two"), (Writable) new Text("three"));
        String header = ",one,two,three";
        List<String> lines = new ArrayList<>();
        for (int i = 0; i < numLines; i++)
            lines.add(Integer.toString(i) + header);
        File tempFile = File.createTempFile("csvSkipLines", ".csv");
        FileUtils.writeLines(tempFile, lines);

        CSVRecordReader rr = new CSVRecordReader(numLines, ',');
        rr.initialize(new FileSplit(tempFile));
        rr.reset();
        assertTrue(!rr.hasNext());
        rr.next();
    }

    @Test
    public void testCsvSkipAllButOneLine() throws IOException, InterruptedException {
        final int numLines = 4;
        final List<Writable> lineList = Arrays.<Writable>asList(new Text(Integer.toString(numLines - 1)),
                new Text("one"), new Text("two"), new Text("three"));
        String header = ",one,two,three";
        List<String> lines = new ArrayList<>();
        for (int i = 0; i < numLines; i++)
            lines.add(Integer.toString(i) + header);
        File tempFile = File.createTempFile("csvSkipLines", ".csv");
        FileUtils.writeLines(tempFile, lines);

        CSVRecordReader rr = new CSVRecordReader(numLines - 1, ',');
        rr.initialize(new FileSplit(tempFile));
        rr.reset();
        assertTrue(rr.hasNext());
        assertEquals(rr.next(), lineList);
    }


    @Test
    public void testStreamReset() throws Exception {
        CSVRecordReader rr = new CSVRecordReader(0, ',');
        rr.initialize(new InputStreamInputSplit(new ClassPathResource("iris.dat").getInputStream()));

        int count = 0;
        while(rr.hasNext()){
            assertNotNull(rr.next());
            count++;
        }
        assertEquals(150, count);

        assertFalse(rr.resetSupported());

        try{
            rr.reset();
            fail("Expected exception");
        } catch (Exception e){
            e.printStackTrace();
        }
    }
}
