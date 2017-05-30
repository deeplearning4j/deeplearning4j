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

package org.datavec.hadoop.records.reader;

import com.google.common.io.Files;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Text;
import org.datavec.hadoop.records.reader.mapfile.record.SequenceRecordWritable;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.net.URI;
import java.util.*;

import static org.junit.Assert.*;

/**
 * Created by Alex on 29/05/2017.
 */
public class TestMapFileSequenceRecordReader {

    private static File tempDirSeq;
    private static Path seqMapFilePath;
    private static Map<LongWritable,SequenceRecordWritable> seqMap;

    @BeforeClass
    public static void buildMapFiles() throws IOException {

        Configuration c = new Configuration();
        Class<? extends WritableComparable> keyClass = LongWritable.class;
        Class<? extends Writable> valueClass = SequenceRecordWritable.class;

        SequenceFile.Writer.Option[] opts = new SequenceFile.Writer.Option[]{
                MapFile.Writer.keyClass(keyClass),
                SequenceFile.Writer.valueClass(valueClass)
        };

        tempDirSeq = Files.createTempDir();
        seqMapFilePath = new Path("file:///" + tempDirSeq.getAbsolutePath());

        MapFile.Writer writer = new MapFile.Writer(c, seqMapFilePath, opts);

        seqMap = new HashMap<>();
        seqMap.put(new LongWritable(0), new SequenceRecordWritable(
                Arrays.asList(
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("zero"), new IntWritable(0), new DoubleWritable(0)),
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("one"), new IntWritable(1), new DoubleWritable(1.0)),
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("two"), new IntWritable(2), new DoubleWritable(2.0)))
        ));

        seqMap.put(new LongWritable(1), new SequenceRecordWritable(
                Arrays.asList(
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("Bzero"), new IntWritable(10), new DoubleWritable(10)),
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("Bone"), new IntWritable(11), new DoubleWritable(11.0)),
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("Btwo"), new IntWritable(12), new DoubleWritable(12.0)))
        ));

        seqMap.put(new LongWritable(2), new SequenceRecordWritable(
                Arrays.asList(
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("Czero"), new IntWritable(20), new DoubleWritable(20)),
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("Cone"), new IntWritable(21), new DoubleWritable(21.0)),
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("Ctwo"), new IntWritable(22), new DoubleWritable(22.0)))
        ));


        //Need to write in order
        for( int i=0; i<=2; i++ ){
            LongWritable key = new LongWritable(i);
            SequenceRecordWritable value = seqMap.get(key);

            writer.append(key, value);
        }

        writer.close();


    }

    @AfterClass
    public static void destroyMapFiles(){
        tempDirSeq.delete();
        tempDirSeq = null;
        seqMapFilePath = null;
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
        while(seqRR.hasNext()){
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
        int[] order = (int[])f.get(seqRR);
        assertNotNull(order);
        int[] expOrder = new int[]{1,2,0};  //Fixed RNG seed -> always this order
        assertArrayEquals(expOrder, order);

        count = 0;
        while(seqRR.hasNext()){
            List<List<org.datavec.api.writable.Writable>> l = seqRR.sequenceRecord();
            assertEquals(seqMap.get(new LongWritable(expOrder[count])).getSequenceRecord(), l);
            count++;
        }
    }

}
