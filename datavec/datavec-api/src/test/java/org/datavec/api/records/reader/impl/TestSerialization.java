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

package org.datavec.api.records.reader.impl;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.*;
import org.datavec.api.records.reader.impl.jackson.FieldSelection;
import org.datavec.api.records.reader.impl.jackson.JacksonLineRecordReader;
import org.datavec.api.records.reader.impl.jackson.JacksonRecordReader;
import org.datavec.api.records.reader.impl.misc.LibSvmRecordReader;
import org.datavec.api.records.reader.impl.misc.SVMLightRecordReader;
import org.datavec.api.records.reader.impl.regex.RegexLineRecordReader;
import org.datavec.api.records.reader.impl.regex.RegexSequenceRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.MathFunction;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Record readers need to be serializable for spark.
 * Note however that not all are used/usable with spark (such as Collection[Sequence]RecordReader
 * and the rest are generally used without being initialized on a particular dataset
 */
public class TestSerialization {

    @Test
    public void testRR() throws Exception {

        List<RecordReader> rrs = new ArrayList<>();

        rrs.add(new CSVNLinesSequenceRecordReader(10));
        rrs.add(new CSVRecordReader(10, ','));
        rrs.add(new CSVSequenceRecordReader(1, ","));
        rrs.add(new CSVVariableSlidingWindowRecordReader(5));
        rrs.add(new CSVRegexRecordReader(0, ",", null, new String[] {null, "(.+) (.+) (.+)"}));
        rrs.add(new JacksonRecordReader(new FieldSelection.Builder().addField("a").addField(new Text("MISSING_B"), "b")
                .addField(new Text("MISSING_CX"), "c", "x").build(), new ObjectMapper(new JsonFactory())));
        rrs.add(new JacksonLineRecordReader(new FieldSelection.Builder().addField("value1")
        		.addField("value2").build(), new ObjectMapper(new JsonFactory())));
        rrs.add(new LibSvmRecordReader());
        rrs.add(new SVMLightRecordReader());
        rrs.add(new RegexLineRecordReader("(.+) (.+) (.+)", 0));
        rrs.add(new RegexSequenceRecordReader("(.+) (.+) (.+)", 0));
        rrs.add(new TransformProcessRecordReader(new CSVRecordReader(), getTp()));
        rrs.add(new TransformProcessSequenceRecordReader(new CSVSequenceRecordReader(), getTp()));
        rrs.add(new LineRecordReader());

        for(RecordReader r : rrs){
            System.out.println(r.getClass().getName());
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream os = new ObjectOutputStream(baos);
            os.writeObject(r);
            byte[] bytes = baos.toByteArray();

            ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(bytes));

            RecordReader r2 = (RecordReader) ois.readObject();
        }
    }

    private static TransformProcess getTp(){
        Schema s = new Schema.Builder().addColumnDouble("d").build();
        TransformProcess tp = new TransformProcess.Builder(s)
                .doubleMathFunction("d", MathFunction.ABS)
                .build();
        return tp;
    }

    @Test
    public void testCsvRRSerializationResults() throws Exception {
        int skipLines = 3;
        RecordReader r1 = new CSVRecordReader(skipLines, '\t');
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream os = new ObjectOutputStream(baos);
        os.writeObject(r1);
        byte[] bytes = baos.toByteArray();
        ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(bytes));
        RecordReader r2 = (RecordReader) ois.readObject();

        File f = new ClassPathResource("datavec-api/iris_tab_delim.txt").getFile();

        r1.initialize(new FileSplit(f));
        r2.initialize(new FileSplit(f));

        int count = 0;
        while(r1.hasNext()){
            List<Writable> n1 = r1.next();
            List<Writable> n2 = r2.next();
            assertEquals(n1, n2);
            count++;
        }

        assertEquals(150-skipLines, count);
    }

}
