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

package org.datavec.hadoop.records.reader;

import com.google.common.io.Files;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Text;
import org.datavec.hadoop.records.reader.mapfile.MapFileRecordReader;
import org.datavec.hadoop.records.reader.mapfile.MapFileSequenceRecordReader;
import org.datavec.hadoop.records.reader.mapfile.record.RecordWritable;
import org.datavec.hadoop.records.reader.mapfile.record.SequenceRecordWritable;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.MathUtils;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.net.URI;
import java.util.*;

import static org.junit.Assert.*;

/**
 * Created by Alex on 29/05/2017.
 */
public class TestMapFileRecordReader {

    private static File tempDirSeq;
    private static File tempDir;
    private static Path seqMapFilePath;
    private static Path mapFilePath;
    private static Map<LongWritable, SequenceRecordWritable> seqMap;
    private static Map<LongWritable, RecordWritable> recordMap;

    @BeforeClass
    public static void buildMapFiles() throws IOException {

        //----- Sequence RR setup -----

        Configuration c = new Configuration();
        Class<? extends WritableComparable> keyClass = LongWritable.class;
        Class<? extends Writable> valueClass = SequenceRecordWritable.class;

        SequenceFile.Writer.Option[] opts = new SequenceFile.Writer.Option[] {MapFile.Writer.keyClass(keyClass),
                        SequenceFile.Writer.valueClass(valueClass)};

        tempDirSeq = Files.createTempDir();
        seqMapFilePath = new Path("file:///" + tempDirSeq.getAbsolutePath());

        MapFile.Writer writer = new MapFile.Writer(c, seqMapFilePath, opts);

        seqMap = new HashMap<>();
        seqMap.put(new LongWritable(0), new SequenceRecordWritable(Arrays.asList(
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("zero"), new IntWritable(0),
                                        new DoubleWritable(0), new NDArrayWritable(Nd4j.valueArrayOf(10, 0.0))),
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("one"), new IntWritable(1),
                                        new DoubleWritable(1.0), new NDArrayWritable(Nd4j.valueArrayOf(10, 1.0))),
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("two"), new IntWritable(2),
                                        new DoubleWritable(2.0), new NDArrayWritable(Nd4j.valueArrayOf(10, 2.0))))));

        seqMap.put(new LongWritable(1), new SequenceRecordWritable(Arrays.asList(
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("Bzero"), new IntWritable(10),
                                        new DoubleWritable(10), new NDArrayWritable(Nd4j.valueArrayOf(10, 10.0))),
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("Bone"), new IntWritable(11),
                                        new DoubleWritable(11.0), new NDArrayWritable(Nd4j.valueArrayOf(10, 11.0))),
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("Btwo"), new IntWritable(12),
                                        new DoubleWritable(12.0), new NDArrayWritable(Nd4j.valueArrayOf(10, 12.0))))));

        seqMap.put(new LongWritable(2), new SequenceRecordWritable(Arrays.asList(
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("Czero"), new IntWritable(20),
                                        new DoubleWritable(20), new NDArrayWritable(Nd4j.valueArrayOf(10, 20.0))),
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("Cone"), new IntWritable(21),
                                        new DoubleWritable(21.0), new NDArrayWritable(Nd4j.valueArrayOf(10, 21.0))),
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("Ctwo"), new IntWritable(22),
                                        new DoubleWritable(22.0), new NDArrayWritable(Nd4j.valueArrayOf(10, 22.0))))));


        //Need to write in order
        for (int i = 0; i <= 2; i++) {
            LongWritable key = new LongWritable(i);
            SequenceRecordWritable value = seqMap.get(key);

            writer.append(key, value);
        }
        writer.close();


        //----- Standard RR setup -----

        valueClass = RecordWritable.class;

        opts = new SequenceFile.Writer.Option[] {MapFile.Writer.keyClass(keyClass),
                        SequenceFile.Writer.valueClass(valueClass)};

        tempDir = Files.createTempDir();
        mapFilePath = new Path("file:///" + tempDir.getAbsolutePath());

        writer = new MapFile.Writer(c, mapFilePath, opts);

        recordMap = new HashMap<>();
        recordMap.put(new LongWritable(0),
                        new RecordWritable(Arrays.<org.datavec.api.writable.Writable>asList(new Text("zero"),
                                        new IntWritable(0), new DoubleWritable(0),
                                        new NDArrayWritable(Nd4j.valueArrayOf(10, 0.0)))));

        recordMap.put(new LongWritable(1),
                        new RecordWritable(Arrays.<org.datavec.api.writable.Writable>asList(new Text("one"),
                                        new IntWritable(11), new DoubleWritable(11.0),
                                        new NDArrayWritable(Nd4j.valueArrayOf(10, 11.0)))));

        recordMap.put(new LongWritable(2),
                        new RecordWritable(Arrays.<org.datavec.api.writable.Writable>asList(new Text("two"),
                                        new IntWritable(22), new DoubleWritable(22.0),
                                        new NDArrayWritable(Nd4j.valueArrayOf(10, 22.0)))));


        //Need to write in order
        for (int i = 0; i <= 2; i++) {
            LongWritable key = new LongWritable(i);
            RecordWritable value = recordMap.get(key);

            writer.append(key, value);
        }
        writer.close();

    }

    @AfterClass
    public static void destroyMapFiles() {
        tempDirSeq.delete();
        tempDirSeq = null;
        seqMapFilePath = null;
        seqMap = null;

        tempDir.delete();
        tempDir = null;
        mapFilePath = null;
        seqMap = null;
    }

    @Test
    public void testSequenceRecordReader() throws Exception {
        SequenceRecordReader seqRR = new MapFileSequenceRecordReader();
        URI uri = seqMapFilePath.toUri();
        InputSplit is = new FileSplit(new File(uri));
        seqRR.initialize(is);

        assertTrue(seqRR.hasNext());
        int count = 0;
        while (seqRR.hasNext()) {
            List<List<org.datavec.api.writable.Writable>> l = seqRR.sequenceRecord();

            assertEquals(seqMap.get(new LongWritable(count)).getSequenceRecord(), l);

            count++;
        }
        assertEquals(seqMap.size(), count);

        seqRR.close();

        //Try the same thing, but with random order
        seqRR = new MapFileSequenceRecordReader(new Random(12345));
        seqRR.initialize(is);

        Field f = MapFileSequenceRecordReader.class.getDeclaredField("order");
        f.setAccessible(true);
        int[] order = (int[]) f.get(seqRR);
        assertNotNull(order);
        int[] expOrder = new int[]{0,1,2};
        MathUtils.shuffleArray(expOrder, new Random(12345));
        assertArrayEquals(expOrder, order);

        count = 0;
        while (seqRR.hasNext()) {
            List<List<org.datavec.api.writable.Writable>> l = seqRR.sequenceRecord();
            assertEquals(seqMap.get(new LongWritable(expOrder[count])).getSequenceRecord(), l);
            count++;
        }
    }

    @Test
    public void testRecordReader() throws Exception {
        RecordReader rr = new MapFileRecordReader();
        URI uri = mapFilePath.toUri();
        InputSplit is = new FileSplit(new File(uri));
        rr.initialize(is);

        assertTrue(rr.hasNext());
        int count = 0;
        while (rr.hasNext()) {
            List<org.datavec.api.writable.Writable> l = rr.next();

            assertEquals(recordMap.get(new LongWritable(count)).getRecord(), l);

            count++;
        }
        assertEquals(recordMap.size(), count);

        rr.close();

        //Try the same thing, but with random order
        rr = new MapFileRecordReader(new Random(12345));
        rr.initialize(is);

        Field f = MapFileRecordReader.class.getDeclaredField("order");
        f.setAccessible(true);
        int[] order = (int[]) f.get(rr);
        assertNotNull(order);

        int[] expOrder = new int[]{0,1,2};
        MathUtils.shuffleArray(expOrder, new Random(12345));
        assertArrayEquals(expOrder, order);

        count = 0;
        while (rr.hasNext()) {
            List<org.datavec.api.writable.Writable> l = rr.next();
            assertEquals(recordMap.get(new LongWritable(expOrder[count])).getRecord(), l);
            count++;
        }
    }
}
