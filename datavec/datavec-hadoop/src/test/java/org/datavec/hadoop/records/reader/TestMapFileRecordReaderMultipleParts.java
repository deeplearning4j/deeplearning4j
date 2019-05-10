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
import org.datavec.api.writable.Text;
import org.datavec.hadoop.records.reader.mapfile.IndexToKey;
import org.datavec.hadoop.records.reader.mapfile.MapFileRecordReader;
import org.datavec.hadoop.records.reader.mapfile.MapFileSequenceRecordReader;
import org.datavec.hadoop.records.reader.mapfile.index.LongIndexToKey;
import org.datavec.hadoop.records.reader.mapfile.record.RecordWritable;
import org.datavec.hadoop.records.reader.mapfile.record.SequenceRecordWritable;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.MathUtils;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.net.URI;
import java.util.*;

import static org.junit.Assert.*;

/**
 * Basically the same as TestMapfileRecordReader, but we have multiple parts as per say a Spark save operation
 * Paths are like
 * /part-r-00000/data
 * /part-r-00000/index
 * /part-r-00001/data
 * /part-r-00001/index
 * /part-r-00002/data
 * /part-r-00002/index
 */
public class TestMapFileRecordReaderMultipleParts {

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
        File[] subdirs = new File[3];
        Path[] paths = new Path[subdirs.length];
        MapFile.Writer[] writers = new MapFile.Writer[subdirs.length];
        for (int i = 0; i < subdirs.length; i++) {
            subdirs[i] = new File(tempDirSeq, "part-r-0000" + i);
            subdirs[i].mkdir();
            paths[i] = new Path("file:///" + subdirs[i].getAbsolutePath());
            writers[i] = new MapFile.Writer(c, paths[i], opts);
        }
        seqMapFilePath = new Path("file:///" + tempDirSeq.getAbsolutePath());



        seqMap = new HashMap<>();

        for (int i = 0; i < 9; i++) {
            seqMap.put(new LongWritable(i), new SequenceRecordWritable(Arrays.asList(
                            Arrays.<org.datavec.api.writable.Writable>asList(new Text(i + "-0"), new IntWritable(3 * i),
                                            new DoubleWritable(3 * i)),
                            Arrays.<org.datavec.api.writable.Writable>asList(new Text(i + "-1"),
                                            new IntWritable(3 * i + 1), new DoubleWritable(3 * i + 1.0)),
                            Arrays.<org.datavec.api.writable.Writable>asList(new Text(i + "-2"),
                                            new IntWritable(3 * i + 2), new DoubleWritable(3 * i + 2.0)))));
        }


        //Need to write in order, to different map files separately
        for (int i = 0; i < seqMap.size(); i++) {
            int mapFileIdx = i / writers.length;

            LongWritable key = new LongWritable(i);
            SequenceRecordWritable value = seqMap.get(key);

            writers[mapFileIdx].append(key, value);
        }

        for (MapFile.Writer m : writers) {
            m.close();
        }


        //----- Standard RR setup -----

        valueClass = RecordWritable.class;

        opts = new SequenceFile.Writer.Option[] {MapFile.Writer.keyClass(keyClass),
                        SequenceFile.Writer.valueClass(valueClass)};

        tempDir = Files.createTempDir();
        subdirs = new File[3];
        paths = new Path[subdirs.length];
        writers = new MapFile.Writer[subdirs.length];
        for (int i = 0; i < subdirs.length; i++) {
            subdirs[i] = new File(tempDir, "part-r-0000" + i);
            subdirs[i].mkdir();
            paths[i] = new Path("file:///" + subdirs[i].getAbsolutePath());
            writers[i] = new MapFile.Writer(c, paths[i], opts);
        }
        mapFilePath = new Path("file:///" + tempDir.getAbsolutePath());

        recordMap = new HashMap<>();
        for (int i = 0; i < 9; i++) {
            recordMap.put(new LongWritable(i), new RecordWritable(Arrays.<org.datavec.api.writable.Writable>asList(
                            new Text(String.valueOf(i)), new IntWritable(i), new DoubleWritable(i))));
        }


        //Need to write in order
        for (int i = 0; i < recordMap.size(); i++) {
            int mapFileIdx = i / writers.length;
            LongWritable key = new LongWritable(i);
            RecordWritable value = recordMap.get(key);

            writers[mapFileIdx].append(key, value);
        }

        for (MapFile.Writer m : writers) {
            m.close();
        }

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

        //Check number of records calculation
        Field f = MapFileSequenceRecordReader.class.getDeclaredField("indexToKey");
        f.setAccessible(true);
        IndexToKey itk = (IndexToKey) f.get(seqRR);
        assertEquals(seqMap.size(), itk.getNumRecords());

        //Check indices for each map file
        List<Pair<Long, Long>> expReaderExampleIdxs = new ArrayList<>();
        expReaderExampleIdxs.add(new Pair<>(0L, 2L));
        expReaderExampleIdxs.add(new Pair<>(3L, 5L));
        expReaderExampleIdxs.add(new Pair<>(6L, 8L));

        f = LongIndexToKey.class.getDeclaredField("readerIndices");
        f.setAccessible(true);
        assertEquals(expReaderExampleIdxs, f.get(itk));
        //        System.out.println(f.get(itk));

        //Check standard iteration order (no randomization)
        assertTrue(seqRR.hasNext());
        int count = 0;
        while (seqRR.hasNext()) {
            List<List<org.datavec.api.writable.Writable>> l = seqRR.sequenceRecord();

            assertEquals(seqMap.get(new LongWritable(count)).getSequenceRecord(), l);

            count++;
        }
        assertFalse(seqRR.hasNext());
        assertEquals(seqMap.size(), count);

        seqRR.close();

        //Try the same thing, but with random order
        seqRR = new MapFileSequenceRecordReader(new Random(12345));
        seqRR.initialize(is);

        //Check order is defined and as expected
        f = MapFileSequenceRecordReader.class.getDeclaredField("order");
        f.setAccessible(true);
        int[] order = (int[]) f.get(seqRR);
        assertNotNull(order);

        int[] expOrder = new int[9];
        for (int i = 0; i < expOrder.length; i++) {
            expOrder[i] = i;
        }
        MathUtils.shuffleArray(expOrder, new Random(12345));
        assertArrayEquals(expOrder, order);
        //        System.out.println(Arrays.toString(expOrder));

        count = 0;
        while (seqRR.hasNext()) {
            List<List<org.datavec.api.writable.Writable>> l = seqRR.sequenceRecord();
            assertEquals(seqMap.get(new LongWritable(expOrder[count])).getSequenceRecord(), l);
            count++;
        }
    }

    @Test
    public void testRecordReaderMultipleParts() throws Exception {
        RecordReader rr = new MapFileRecordReader();
        URI uri = mapFilePath.toUri();
        InputSplit is = new FileSplit(new File(uri));
        rr.initialize(is);

        //Check number of records calculation
        Field f = MapFileRecordReader.class.getDeclaredField("indexToKey");
        f.setAccessible(true);
        IndexToKey itk = (IndexToKey) f.get(rr);
        assertEquals(seqMap.size(), itk.getNumRecords());

        //Check indices for each map file
        List<Pair<Long, Long>> expReaderExampleIdxs = new ArrayList<>();
        expReaderExampleIdxs.add(new Pair<>(0L, 2L));
        expReaderExampleIdxs.add(new Pair<>(3L, 5L));
        expReaderExampleIdxs.add(new Pair<>(6L, 8L));

        f = LongIndexToKey.class.getDeclaredField("readerIndices");
        f.setAccessible(true);
        assertEquals(expReaderExampleIdxs, f.get(itk));

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

        f = MapFileRecordReader.class.getDeclaredField("order");
        f.setAccessible(true);
        int[] order = (int[]) f.get(rr);
        assertNotNull(order);
        int[] expOrder = new int[9];
        for (int i = 0; i < expOrder.length; i++) {
            expOrder[i] = i;
        }
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
