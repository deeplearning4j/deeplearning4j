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

package org.datavec.api.records.reader.impl.transform;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.inmemory.InMemorySequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.Writable;
import org.joda.time.DateTimeZone;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by agibsonccc on 3/21/17.
 */
public class TransformProcessRecordReaderTests {

    @Test
    public void simpleTransformTest() throws Exception {
        Schema schema = new Schema.Builder()
                .addColumnsDouble("%d", 0, 4)
                .build();
        TransformProcess transformProcess = new TransformProcess.Builder(schema).removeColumns("0").build();
        CSVRecordReader csvRecordReader = new CSVRecordReader();
        csvRecordReader.initialize(new FileSplit(new ClassPathResource("datavec-api/iris.dat").getFile()));
        TransformProcessRecordReader rr =
                        new TransformProcessRecordReader(csvRecordReader, transformProcess);
        int count = 0;
        List<List<Writable>> all = new ArrayList<>();
        while(rr.hasNext()){
            List<Writable> next = rr.next();
            assertEquals(4, next.size());
            count++;
            all.add(next);
        }
        assertEquals(150, count);

        //Test batch:
        assertTrue(rr.resetSupported());
        rr.reset();
        List<List<Writable>> batch = rr.next(150);
        assertEquals(all, batch);
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
        TransformProcessSequenceRecordReader transformProcessSequenceRecordReader =
                        new TransformProcessSequenceRecordReader(inMemorySequenceRecordReader, transformProcess);
        List<List<Writable>> next = transformProcessSequenceRecordReader.sequenceRecord();
        assertEquals(2, next.get(0).size());

    }
}
