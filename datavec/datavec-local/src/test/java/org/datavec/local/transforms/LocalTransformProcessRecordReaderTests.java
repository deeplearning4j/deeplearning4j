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

package org.datavec.local.transforms;

import org.datavec.api.records.Record;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.inmemory.InMemorySequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.CategoricalColumnCondition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.joda.time.DateTimeZone;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class LocalTransformProcessRecordReaderTests {

    @Test
    public void simpleTransformTest() throws Exception {
        Schema schema = new Schema.Builder().addColumnDouble("0").addColumnDouble("1").addColumnDouble("2")
                .addColumnDouble("3").addColumnDouble("4").build();
        TransformProcess transformProcess = new TransformProcess.Builder(schema).removeColumns("0").build();
        CSVRecordReader csvRecordReader = new CSVRecordReader();
        csvRecordReader.initialize(new FileSplit(new ClassPathResource("iris.dat").getFile()));
        LocalTransformProcessRecordReader transformProcessRecordReader =
                new LocalTransformProcessRecordReader(csvRecordReader, transformProcess);
        assertEquals(4, transformProcessRecordReader.next().size());

    }

    @Test
    public void simpleTransformTestSequence() {
        List<List<Writable>> sequence = new ArrayList<>();
        //First window:
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L), new IntWritable(0),
                new IntWritable(0)));
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L + 100L), new IntWritable(1),
                new IntWritable(0)));
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L + 200L), new IntWritable(2),
                new IntWritable(0)));

        Schema schema = new SequenceSchema.Builder().addColumnTime("timecolumn", DateTimeZone.UTC)
                .addColumnInteger("intcolumn").addColumnInteger("intcolumn2").build();
        TransformProcess transformProcess = new TransformProcess.Builder(schema).removeColumns("intcolumn2").build();
        InMemorySequenceRecordReader inMemorySequenceRecordReader =
                new InMemorySequenceRecordReader(Arrays.asList(sequence));
        LocalTransformProcessSequenceRecordReader transformProcessSequenceRecordReader =
                new LocalTransformProcessSequenceRecordReader(inMemorySequenceRecordReader, transformProcess);
        List<List<Writable>> next = transformProcessSequenceRecordReader.sequenceRecord();
        assertEquals(2, next.get(0).size());

    }


    @Test
    public void testLocalFilter(){

        List<List<Writable>> in = new ArrayList<>();
        in.add(Arrays.asList(new Text("Keep"), new IntWritable(0)));
        in.add(Arrays.asList(new Text("Remove"), new IntWritable(1)));
        in.add(Arrays.asList(new Text("Keep"), new IntWritable(2)));
        in.add(Arrays.asList(new Text("Remove"), new IntWritable(3)));

        Schema s = new Schema.Builder()
                .addColumnCategorical("cat", "Keep", "Remove")
                .addColumnInteger("int")
                .build();

        TransformProcess tp = new TransformProcess.Builder(s)
                .filter(new CategoricalColumnCondition("cat", ConditionOp.Equal, "Remove"))
                .build();

        RecordReader rr = new CollectionRecordReader(in);
        LocalTransformProcessRecordReader ltprr = new LocalTransformProcessRecordReader(rr, tp);

        List<List<Writable>> out = new ArrayList<>();
        while(ltprr.hasNext()){
            out.add(ltprr.next());
        }

        List<List<Writable>> exp = Arrays.asList(in.get(0), in.get(2));

        assertEquals(exp, out);

        //Check reset:
        ltprr.reset();
        out.clear();
        while(ltprr.hasNext()){
            out.add(ltprr.next());
        }
        assertEquals(exp, out);


        //Also test Record method:
        List<Record> rl = new ArrayList<>();
        rr.reset();
        while(rr.hasNext()){
            rl.add(rr.nextRecord());
        }
        List<Record> exp2 = Arrays.asList(rl.get(0), rl.get(2));

        List<Record> act = new ArrayList<>();
        ltprr.reset();
        while(ltprr.hasNext()){
            act.add(ltprr.nextRecord());
        }
    }

}
