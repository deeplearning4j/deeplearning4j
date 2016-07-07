/*
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
import org.datavec.api.io.data.IntWritable;
import org.datavec.api.io.data.Text;
import org.datavec.api.records.writer.impl.CSVRecordWriter;
import org.datavec.api.records.writer.impl.FileRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.StringSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.junit.Test;

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
