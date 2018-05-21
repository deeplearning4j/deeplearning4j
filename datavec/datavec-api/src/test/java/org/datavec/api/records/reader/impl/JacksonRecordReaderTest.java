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

import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.jackson.FieldSelection;
import org.datavec.api.records.reader.impl.jackson.JacksonRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.dataformat.xml.XmlFactory;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;

import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * Created by Alex on 11/04/2016.
 */
public class JacksonRecordReaderTest {

    @Test
    public void testReadingJson() throws Exception {
        //Load 3 values from 3 JSON files
        //stricture: a:value, b:value, c:x:value, c:y:value
        //And we want to load only a:value, b:value and c:x:value
        //For first JSON file: all values are present
        //For second JSON file: b:value is missing
        //For third JSON file: c:x:value is missing

        ClassPathResource cpr = new ClassPathResource("json/json_test_0.txt");
        String path = cpr.getFile().getAbsolutePath().replace("0", "%d");

        InputSplit is = new NumberedFileInputSplit(path, 0, 2);

        RecordReader rr = new JacksonRecordReader(getFieldSelection(), new ObjectMapper(new JsonFactory()));
        rr.initialize(is);

        testJacksonRecordReader(rr);
    }

    @Test
    public void testReadingYaml() throws Exception {
        //Exact same information as JSON format, but in YAML format

        ClassPathResource cpr = new ClassPathResource("yaml/yaml_test_0.txt");
        String path = cpr.getFile().getAbsolutePath().replace("0", "%d");

        InputSplit is = new NumberedFileInputSplit(path, 0, 2);

        RecordReader rr = new JacksonRecordReader(getFieldSelection(), new ObjectMapper(new YAMLFactory()));
        rr.initialize(is);

        testJacksonRecordReader(rr);
    }

    @Test
    public void testReadingXml() throws Exception {
        //Exact same information as JSON format, but in XML format

        ClassPathResource cpr = new ClassPathResource("xml/xml_test_0.txt");
        String path = cpr.getFile().getAbsolutePath().replace("0", "%d");

        InputSplit is = new NumberedFileInputSplit(path, 0, 2);

        RecordReader rr = new JacksonRecordReader(getFieldSelection(), new ObjectMapper(new XmlFactory()));
        rr.initialize(is);

        testJacksonRecordReader(rr);
    }


    private static FieldSelection getFieldSelection() {
        return new FieldSelection.Builder().addField("a").addField(new Text("MISSING_B"), "b")
                        .addField(new Text("MISSING_CX"), "c", "x").build();
    }



    private static void testJacksonRecordReader(RecordReader rr) {

        List<Writable> json0 = rr.next();
        List<Writable> exp0 = Arrays.asList((Writable) new Text("aValue0"), new Text("bValue0"), new Text("cxValue0"));
        assertEquals(exp0, json0);

        List<Writable> json1 = rr.next();
        List<Writable> exp1 =
                        Arrays.asList((Writable) new Text("aValue1"), new Text("MISSING_B"), new Text("cxValue1"));
        assertEquals(exp1, json1);

        List<Writable> json2 = rr.next();
        List<Writable> exp2 =
                        Arrays.asList((Writable) new Text("aValue2"), new Text("bValue2"), new Text("MISSING_CX"));
        assertEquals(exp2, json2);

        assertFalse(rr.hasNext());

        //Test reset
        rr.reset();
        assertEquals(exp0, rr.next());
        assertEquals(exp1, rr.next());
        assertEquals(exp2, rr.next());
        assertFalse(rr.hasNext());
    }

    @Test
    public void testAppendingLabels() throws Exception {
        ClassPathResource cpr = new ClassPathResource("json/json_test_0.txt");
        String path = cpr.getFile().getAbsolutePath().replace("0", "%d");

        InputSplit is = new NumberedFileInputSplit(path, 0, 2);

        //Insert at the end:
        RecordReader rr = new JacksonRecordReader(getFieldSelection(), new ObjectMapper(new JsonFactory()), false, -1,
                        new LabelGen());
        rr.initialize(is);

        List<Writable> exp0 = Arrays.asList((Writable) new Text("aValue0"), new Text("bValue0"), new Text("cxValue0"),
                        new IntWritable(0));
        assertEquals(exp0, rr.next());

        List<Writable> exp1 = Arrays.asList((Writable) new Text("aValue1"), new Text("MISSING_B"), new Text("cxValue1"),
                        new IntWritable(1));
        assertEquals(exp1, rr.next());

        List<Writable> exp2 = Arrays.asList((Writable) new Text("aValue2"), new Text("bValue2"), new Text("MISSING_CX"),
                        new IntWritable(2));
        assertEquals(exp2, rr.next());

        //Insert at position 0:
        rr = new JacksonRecordReader(getFieldSelection(), new ObjectMapper(new JsonFactory()), false, -1,
                        new LabelGen(), 0);
        rr.initialize(is);

        exp0 = Arrays.asList((Writable) new IntWritable(0), new Text("aValue0"), new Text("bValue0"),
                        new Text("cxValue0"));
        assertEquals(exp0, rr.next());

        exp1 = Arrays.asList((Writable) new IntWritable(1), new Text("aValue1"), new Text("MISSING_B"),
                        new Text("cxValue1"));
        assertEquals(exp1, rr.next());

        exp2 = Arrays.asList((Writable) new IntWritable(2), new Text("aValue2"), new Text("bValue2"),
                        new Text("MISSING_CX"));
        assertEquals(exp2, rr.next());
    }

    @Test
    public void testAppendingLabelsMetaData() throws Exception {
        ClassPathResource cpr = new ClassPathResource("json/json_test_0.txt");
        String path = cpr.getFile().getAbsolutePath().replace("0", "%d");

        InputSplit is = new NumberedFileInputSplit(path, 0, 2);

        //Insert at the end:
        RecordReader rr = new JacksonRecordReader(getFieldSelection(), new ObjectMapper(new JsonFactory()), false, -1,
                        new LabelGen());
        rr.initialize(is);

        List<List<Writable>> out = new ArrayList<>();
        while (rr.hasNext()) {
            out.add(rr.next());
        }
        assertEquals(3, out.size());

        rr.reset();

        List<List<Writable>> out2 = new ArrayList<>();
        List<Record> outRecord = new ArrayList<>();
        List<RecordMetaData> meta = new ArrayList<>();
        while (rr.hasNext()) {
            Record r = rr.nextRecord();
            out2.add(r.getRecord());
            outRecord.add(r);
            meta.add(r.getMetaData());
        }

        assertEquals(out, out2);

        List<Record> fromMeta = rr.loadFromMetaData(meta);
        assertEquals(outRecord, fromMeta);
    }


    private static class LabelGen implements PathLabelGenerator {

        @Override
        public Writable getLabelForPath(String path) {
            if (path.endsWith("0.txt"))
                return new IntWritable(0);
            else if (path.endsWith("1.txt"))
                return new IntWritable(1);
            else
                return new IntWritable(2);
        }

        @Override
        public Writable getLabelForPath(URI uri) {
            return getLabelForPath(uri.getPath());
        }

        @Override
        public boolean inferLabelClasses() {
            return true;
        }
    }

}
