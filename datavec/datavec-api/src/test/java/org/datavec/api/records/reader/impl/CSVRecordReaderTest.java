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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.common.io.ClassPathResource;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

import static org.junit.jupiter.api.Assertions.*;


@DisplayName("Csv Record Reader Test")
class CSVRecordReaderTest extends BaseND4JTest {

    @Test
    @DisplayName("Test Next")
    void testNext() throws Exception {
        CSVRecordReader reader = new CSVRecordReader();
        reader.initialize(new StringSplit("1,1,8.0,,,,14.0,,,,15.0,,,,,,,,,,,,1"));
        while (reader.hasNext()) {
            List<Writable> vals = reader.next();
            List<Writable> arr = new ArrayList<>(vals);
            assertEquals(23, vals.size(), "Entry count");
            Text lastEntry = (Text) arr.get(arr.size() - 1);
            assertEquals(1, lastEntry.getLength(), "Last entry garbage");
        }
    }

    @Test
    @DisplayName("Test Empty Entries")
    void testEmptyEntries() throws Exception {
        CSVRecordReader reader = new CSVRecordReader();
        reader.initialize(new StringSplit("1,1,8.0,,,,14.0,,,,15.0,,,,,,,,,,,,"));
        while (reader.hasNext()) {
            List<Writable> vals = reader.next();
            assertEquals(23, vals.size(), "Entry count");
        }
    }

    @Test
    @DisplayName("Test Reset")
    void testReset() throws Exception {
        CSVRecordReader rr = new CSVRecordReader(0, ',');
        rr.initialize(new FileSplit(new ClassPathResource("datavec-api/iris.dat").getFile()));
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
    @DisplayName("Test Reset With Skip Lines")
    void testResetWithSkipLines() throws Exception {
        CSVRecordReader rr = new CSVRecordReader(10, ',');
        rr.initialize(new FileSplit(new ClassPathResource("datavec-api/iris.dat").getFile()));
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
    @DisplayName("Test Write")
    void testWrite() throws Exception {
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
        writer.initialize(fileSplit, new NumberOfRecordsPartitioner());
        for (List<Writable> c : list) {
            writer.write(c);
        }
        writer.close();
        // Read file back in; compare
        String fileContents = FileUtils.readFileToString(p.toFile(), FileRecordWriter.DEFAULT_CHARSET.name());
        // System.out.println(expected);
        // System.out.println("----------");
        // System.out.println(fileContents);
        assertEquals(expected, fileContents);
    }

    @Test
    @DisplayName("Test Tabs As Split 1")
    void testTabsAsSplit1() throws Exception {
        CSVRecordReader reader = new CSVRecordReader(0, '\t');
        reader.initialize(new FileSplit(new ClassPathResource("datavec-api/tabbed.txt").getFile()));
        while (reader.hasNext()) {
            List<Writable> list = new ArrayList<>(reader.next());
            assertEquals(2, list.size());
        }
    }

    @Test
    @DisplayName("Test Pipes As Split")
    void testPipesAsSplit() throws Exception {
        CSVRecordReader reader = new CSVRecordReader(0, '|');
        reader.initialize(new FileSplit(new ClassPathResource("datavec-api/issue414.csv").getFile()));
        int lineidx = 0;
        List<Integer> sixthColumn = Arrays.asList(13, 95, 15, 25);
        while (reader.hasNext()) {
            List<Writable> list = new ArrayList<>(reader.next());
            assertEquals(10, list.size());
            assertEquals((long) sixthColumn.get(lineidx), list.get(5).toInt());
            lineidx++;
        }
    }

    @Test
    @DisplayName("Test With Quotes")
    void testWithQuotes() throws Exception {
        CSVRecordReader reader = new CSVRecordReader(0, ',', '\"');
        reader.initialize(new StringSplit("1,0,3,\"Braund, Mr. Owen Harris\",male,\"\"\"\""));
        while (reader.hasNext()) {
            List<Writable> vals = reader.next();
            assertEquals(6, vals.size(), "Entry count");
            assertEquals(vals.get(0).toString(), "1");
            assertEquals(vals.get(1).toString(), "0");
            assertEquals(vals.get(2).toString(), "3");
            assertEquals(vals.get(3).toString(), "Braund, Mr. Owen Harris");
            assertEquals(vals.get(4).toString(), "male");
            assertEquals(vals.get(5).toString(), "\"");
        }
    }

    @Test
    @DisplayName("Test Meta")
    void testMeta() throws Exception {
        CSVRecordReader rr = new CSVRecordReader(0, ',');
        rr.initialize(new FileSplit(new ClassPathResource("datavec-api/iris.dat").getFile()));
        int lineCount = 0;
        List<RecordMetaData> metaList = new ArrayList<>();
        List<List<Writable>> writables = new ArrayList<>();
        while (rr.hasNext()) {
            Record r = rr.nextRecord();
            assertEquals(5, r.getRecord().size());
            lineCount++;
            RecordMetaData meta = r.getMetaData();
            // System.out.println(r.getRecord() + "\t" + meta.getLocation() + "\t" + meta.getURI());
            metaList.add(meta);
            writables.add(r.getRecord());
        }
        assertFalse(rr.hasNext());
        assertEquals(150, lineCount);
        rr.reset();
        System.out.println("\n\n\n--------------------------------");
        List<Record> contents = rr.loadFromMetaData(metaList);
        assertEquals(150, contents.size());
        // for(Record r : contents ){
        // System.out.println(r);
        // }
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
    @DisplayName("Test Regex")
    void testRegex() throws Exception {
        CSVRecordReader reader = new CSVRegexRecordReader(0, ",", null, new String[] { null, "(.+) (.+) (.+)" });
        reader.initialize(new StringSplit("normal,1.2.3.4 space separator"));
        while (reader.hasNext()) {
            List<Writable> vals = reader.next();
            assertEquals(4, vals.size(), "Entry count");
            assertEquals(vals.get(0).toString(), "normal");
            assertEquals(vals.get(1).toString(), "1.2.3.4");
            assertEquals(vals.get(2).toString(), "space");
            assertEquals(vals.get(3).toString(), "separator");
        }
    }

    @Test
    @DisplayName("Test Csv Skip All Lines")
    void testCsvSkipAllLines() {
        assertThrows(NoSuchElementException.class, () -> {
            final int numLines = 4;
            final List<Writable> lineList = Arrays.asList((Writable) new IntWritable(numLines - 1), (Writable) new Text("one"), (Writable) new Text("two"), (Writable) new Text("three"));
            String header = ",one,two,three";
            List<String> lines = new ArrayList<>();
            for (int i = 0; i < numLines; i++) lines.add(Integer.toString(i) + header);
            File tempFile = File.createTempFile("csvSkipLines", ".csv");
            FileUtils.writeLines(tempFile, lines);
            CSVRecordReader rr = new CSVRecordReader(numLines, ',');
            rr.initialize(new FileSplit(tempFile));
            rr.reset();
            assertTrue(!rr.hasNext());
            rr.next();
        });
    }

    @Test
    @DisplayName("Test Csv Skip All But One Line")
    void testCsvSkipAllButOneLine() throws IOException, InterruptedException {
        final int numLines = 4;
        final List<Writable> lineList = Arrays.<Writable>asList(new Text(Integer.toString(numLines - 1)), new Text("one"), new Text("two"), new Text("three"));
        String header = ",one,two,three";
        List<String> lines = new ArrayList<>();
        for (int i = 0; i < numLines; i++) lines.add(Integer.toString(i) + header);
        File tempFile = File.createTempFile("csvSkipLines", ".csv");
        FileUtils.writeLines(tempFile, lines);
        CSVRecordReader rr = new CSVRecordReader(numLines - 1, ',');
        rr.initialize(new FileSplit(tempFile));
        rr.reset();
        assertTrue(rr.hasNext());
        assertEquals(rr.next(), lineList);
    }

    @Test
    @DisplayName("Test Stream Reset")
    void testStreamReset() throws Exception {
        CSVRecordReader rr = new CSVRecordReader(0, ',');
        rr.initialize(new InputStreamInputSplit(new ClassPathResource("datavec-api/iris.dat").getInputStream()));
        int count = 0;
        while (rr.hasNext()) {
            assertNotNull(rr.next());
            count++;
        }
        assertEquals(150, count);
        assertFalse(rr.resetSupported());
        try {
            rr.reset();
            fail("Expected exception");
        } catch (Exception e) {
            String msg = e.getMessage();
            String msg2 = e.getCause().getMessage();
            assertTrue(msg.contains("Error during LineRecordReader reset"),msg);
            assertTrue(msg2.contains("Reset not supported from streams"),msg2);
            // e.printStackTrace();
        }
    }

    @Test
    @DisplayName("Test Useful Exception No Init")
    void testUsefulExceptionNoInit() {
        CSVRecordReader rr = new CSVRecordReader(0, ',');
        try {
            rr.hasNext();
            fail("Expected exception");
        } catch (Exception e) {
            assertTrue( e.getMessage().contains("initialized"),e.getMessage());
        }
        try {
            rr.next();
            fail("Expected exception");
        } catch (Exception e) {
            assertTrue(e.getMessage().contains("initialized"),e.getMessage());
        }
    }
}
