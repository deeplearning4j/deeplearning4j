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

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.reader.impl.misc.SVMLightRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;
import java.util.*;

import static org.datavec.api.records.reader.impl.misc.SVMLightRecordReader.*;
import static org.junit.Assert.assertEquals;

/**
 * Unit tests for VMLightRecordReader. Replaces reader tests in
 * SVMRecordWriterTest.
 *
 * @see SVMLightRecordReader
 * @see LibSvmTest
 * @see SVMRecordWriterTest
 *
 * @author dave@skymind.io
 */
public class SVMLightRecordReaderTest {

    @Test
    public void testBasicRecord() throws IOException, InterruptedException {
        Map<Integer, List<Writable>> correct = new HashMap<>();
        // 7 2:1 4:2 6:3 8:4 10:5
        correct.put(0, Arrays.asList(ZERO, ONE,
                                    ZERO, new DoubleWritable(2),
                                    ZERO, new DoubleWritable(3),
                                    ZERO, new DoubleWritable(4),
                                    ZERO, new DoubleWritable(5),
                                    new IntWritable(7)));
        // 2 qid:42 1:0.1 2:2 6:6.6 8:80
        correct.put(1, Arrays.asList(new DoubleWritable(0.1), new DoubleWritable(2),
                                    ZERO, ZERO,
                                    ZERO, new DoubleWritable(6.6),
                                    ZERO, new DoubleWritable(80),
                                    ZERO, ZERO,
                                    new IntWritable(2)));
        // 33
        correct.put(2, Arrays.asList(ZERO, ZERO,
                                    ZERO, ZERO,
                                    ZERO, ZERO,
                                    ZERO, ZERO,
                                    ZERO, ZERO,
                                    new IntWritable(33)));

        SVMLightRecordReader rr = new SVMLightRecordReader();
        Configuration config = new Configuration();
        config.setBoolean(SVMLightRecordReader.ZERO_BASED_INDEXING, false);
        config.setInt(SVMLightRecordReader.NUM_FEATURES, 10);
        rr.initialize(config, new FileSplit(new ClassPathResource("datavec-api/svmlight/basic.txt").getFile()));
        int i = 0;
        while (rr.hasNext()) {
            List<Writable> record = rr.next();
            assertEquals(correct.get(i), record);
            i++;
        }
        assertEquals(i, correct.size());
    }

    @Test
    public void testNoAppendLabel() throws IOException, InterruptedException {
        Map<Integer, List<Writable>> correct = new HashMap<>();
        // 7 2:1 4:2 6:3 8:4 10:5
        correct.put(0, Arrays.asList(ZERO, ONE,
                                    ZERO, new DoubleWritable(2),
                                    ZERO, new DoubleWritable(3),
                                    ZERO, new DoubleWritable(4),
                                    ZERO, new DoubleWritable(5)));
        // 2 qid:42 1:0.1 2:2 6:6.6 8:80
        correct.put(1, Arrays.asList(new DoubleWritable(0.1), new DoubleWritable(2),
                                    ZERO, ZERO,
                                    ZERO, new DoubleWritable(6.6),
                                    ZERO, new DoubleWritable(80),
                                    ZERO, ZERO));
        // 33
        correct.put(2, Arrays.asList(ZERO, ZERO,
                                    ZERO, ZERO,
                                    ZERO, ZERO,
                                    ZERO, ZERO,
                                    ZERO, ZERO));

        SVMLightRecordReader rr = new SVMLightRecordReader();
        Configuration config = new Configuration();
        config.setBoolean(SVMLightRecordReader.ZERO_BASED_INDEXING, false);
        config.setInt(SVMLightRecordReader.NUM_FEATURES, 10);
        config.setBoolean(SVMLightRecordReader.APPEND_LABEL, false);
        rr.initialize(config, new FileSplit(new ClassPathResource("datavec-api/svmlight/basic.txt").getFile()));
        int i = 0;
        while (rr.hasNext()) {
            List<Writable> record = rr.next();
            assertEquals(correct.get(i), record);
            i++;
        }
        assertEquals(i, correct.size());
    }

    @Test
    public void testNoLabel() throws IOException, InterruptedException {
        Map<Integer, List<Writable>> correct = new HashMap<>();
        //  2:1 4:2 6:3 8:4 10:5
        correct.put(0, Arrays.asList(ZERO, ONE,
                ZERO, new DoubleWritable(2),
                ZERO, new DoubleWritable(3),
                ZERO, new DoubleWritable(4),
                ZERO, new DoubleWritable(5)));
        //  qid:42 1:0.1 2:2 6:6.6 8:80
        correct.put(1, Arrays.asList(new DoubleWritable(0.1), new DoubleWritable(2),
                ZERO, ZERO,
                ZERO, new DoubleWritable(6.6),
                ZERO, new DoubleWritable(80),
                ZERO, ZERO));
        //  1:1.0
        correct.put(2, Arrays.asList(new DoubleWritable(1.0), ZERO,
                ZERO, ZERO,
                ZERO, ZERO,
                ZERO, ZERO,
                ZERO, ZERO));
        //
        correct.put(3, Arrays.asList(ZERO, ZERO,
                ZERO, ZERO,
                ZERO, ZERO,
                ZERO, ZERO,
                ZERO, ZERO));

        SVMLightRecordReader rr = new SVMLightRecordReader();
        Configuration config = new Configuration();
        config.setBoolean(SVMLightRecordReader.ZERO_BASED_INDEXING, false);
        config.setInt(SVMLightRecordReader.NUM_FEATURES, 10);
        config.setBoolean(SVMLightRecordReader.APPEND_LABEL, true);
        rr.initialize(config, new FileSplit(new ClassPathResource("datavec-api/svmlight/noLabels.txt").getFile()));
        int i = 0;
        while (rr.hasNext()) {
            List<Writable> record = rr.next();
            assertEquals(correct.get(i), record);
            i++;
        }
        assertEquals(i, correct.size());
    }

    @Test
    public void testMultioutputRecord() throws IOException, InterruptedException {
        Map<Integer, List<Writable>> correct = new HashMap<>();
        // 7 2.45,9 2:1 4:2 6:3 8:4 10:5
        correct.put(0, Arrays.asList(ZERO, ONE,
                                    ZERO, new DoubleWritable(2),
                                    ZERO, new DoubleWritable(3),
                                    ZERO, new DoubleWritable(4),
                                    ZERO, new DoubleWritable(5),
                                    new IntWritable(7), new DoubleWritable(2.45),
                                    new IntWritable(9)));
        // 2,3,4 qid:42 1:0.1 2:2 6:6.6 8:80
        correct.put(1, Arrays.asList(new DoubleWritable(0.1), new DoubleWritable(2),
                                    ZERO, ZERO,
                                    ZERO, new DoubleWritable(6.6),
                                    ZERO, new DoubleWritable(80),
                                    ZERO, ZERO,
                                    new IntWritable(2), new IntWritable(3),
                                    new IntWritable(4)));
        // 33,32.0,31.9
        correct.put(2, Arrays.asList(ZERO, ZERO,
                                    ZERO, ZERO,
                                    ZERO, ZERO,
                                    ZERO, ZERO,
                                    ZERO, ZERO,
                                    new IntWritable(33), new DoubleWritable(32.0),
                                    new DoubleWritable(31.9)));

        SVMLightRecordReader rr = new SVMLightRecordReader();
        Configuration config = new Configuration();
        config.setBoolean(SVMLightRecordReader.ZERO_BASED_INDEXING, false);
        config.setInt(SVMLightRecordReader.NUM_FEATURES, 10);
        rr.initialize(config, new FileSplit(new ClassPathResource("datavec-api/svmlight/multioutput.txt").getFile()));
        int i = 0;
        while (rr.hasNext()) {
            List<Writable> record = rr.next();
            assertEquals(correct.get(i), record);
            i++;
        }
        assertEquals(i, correct.size());
    }


    @Test
    public void testMultilabelRecord() throws IOException, InterruptedException {
        Map<Integer, List<Writable>> correct = new HashMap<>();
        // 1,3 2:1 4:2 6:3 8:4 10:5
        correct.put(0, Arrays.asList(ZERO, ONE,
                                    ZERO, new DoubleWritable(2),
                                    ZERO, new DoubleWritable(3),
                                    ZERO, new DoubleWritable(4),
                                    ZERO, new DoubleWritable(5),
                                    LABEL_ONE, LABEL_ZERO,
                                    LABEL_ONE, LABEL_ZERO));
        // 2 qid:42 1:0.1 2:2 6:6.6 8:80
        correct.put(1, Arrays.asList(new DoubleWritable(0.1), new DoubleWritable(2),
                                    ZERO, ZERO,
                                    ZERO, new DoubleWritable(6.6),
                                    ZERO, new DoubleWritable(80),
                                    ZERO, ZERO,
                                    LABEL_ZERO, LABEL_ONE,
                                    LABEL_ZERO, LABEL_ZERO));
        // 1,2,4
        correct.put(2, Arrays.asList(ZERO, ZERO,
                                    ZERO, ZERO,
                                    ZERO, ZERO,
                                    ZERO, ZERO,
                                    ZERO, ZERO,
                                    LABEL_ONE, LABEL_ONE,
                                    LABEL_ZERO, LABEL_ONE));
        //  1:1.0
        correct.put(3, Arrays.asList(new DoubleWritable(1.0), ZERO,
                ZERO, ZERO,
                ZERO, ZERO,
                ZERO, ZERO,
                ZERO, ZERO,
                LABEL_ZERO, LABEL_ZERO,
                LABEL_ZERO, LABEL_ZERO));
        //
        correct.put(4, Arrays.asList(ZERO, ZERO,
                ZERO, ZERO,
                ZERO, ZERO,
                ZERO, ZERO,
                ZERO, ZERO,
                LABEL_ZERO, LABEL_ZERO,
                LABEL_ZERO, LABEL_ZERO));

        SVMLightRecordReader rr = new SVMLightRecordReader();
        Configuration config = new Configuration();
        config.setBoolean(SVMLightRecordReader.ZERO_BASED_INDEXING, false);
        config.setInt(SVMLightRecordReader.NUM_FEATURES, 10);
        config.setBoolean(SVMLightRecordReader.MULTILABEL, true);
        config.setInt(SVMLightRecordReader.NUM_LABELS, 4);
        rr.initialize(config, new FileSplit(new ClassPathResource("datavec-api/svmlight/multilabel.txt").getFile()));
        int i = 0;
        while (rr.hasNext()) {
            List<Writable> record = rr.next();
            assertEquals(correct.get(i), record);
            i++;
        }
        assertEquals(i, correct.size());
    }

    @Test
    public void testZeroBasedIndexing() throws IOException, InterruptedException {
        Map<Integer, List<Writable>> correct = new HashMap<>();
        // 1,3 2:1 4:2 6:3 8:4 10:5
        correct.put(0, Arrays.asList(ZERO,
                                    ZERO, ONE,
                                    ZERO, new DoubleWritable(2),
                                    ZERO, new DoubleWritable(3),
                                    ZERO, new DoubleWritable(4),
                                    ZERO, new DoubleWritable(5),
                                    LABEL_ZERO,
                                    LABEL_ONE, LABEL_ZERO,
                                    LABEL_ONE, LABEL_ZERO));
        // 2 qid:42 1:0.1 2:2 6:6.6 8:80
        correct.put(1, Arrays.asList(ZERO,
                                    new DoubleWritable(0.1), new DoubleWritable(2),
                                    ZERO, ZERO,
                                    ZERO, new DoubleWritable(6.6),
                                    ZERO, new DoubleWritable(80),
                                    ZERO, ZERO,
                                    LABEL_ZERO,
                                    LABEL_ZERO, LABEL_ONE,
                                    LABEL_ZERO, LABEL_ZERO));
        // 1,2,4
        correct.put(2, Arrays.asList(ZERO,
                                    ZERO, ZERO,
                                    ZERO, ZERO,
                                    ZERO, ZERO,
                                    ZERO, ZERO,
                                    ZERO, ZERO,
                                    LABEL_ZERO,
                                    LABEL_ONE, LABEL_ONE,
                                    LABEL_ZERO, LABEL_ONE));
        //  1:1.0
        correct.put(3, Arrays.asList(ZERO,
                new DoubleWritable(1.0), ZERO,
                ZERO, ZERO,
                ZERO, ZERO,
                ZERO, ZERO,
                ZERO, ZERO,
                LABEL_ZERO,
                LABEL_ZERO, LABEL_ZERO,
                LABEL_ZERO, LABEL_ZERO));
        //
        correct.put(4, Arrays.asList(ZERO,
                ZERO, ZERO,
                ZERO, ZERO,
                ZERO, ZERO,
                ZERO, ZERO,
                ZERO, ZERO,
                LABEL_ZERO,
                LABEL_ZERO, LABEL_ZERO,
                LABEL_ZERO, LABEL_ZERO));

        SVMLightRecordReader rr = new SVMLightRecordReader();
        Configuration config = new Configuration();
        // Zero-based indexing is default
        config.setBoolean(SVMLightRecordReader.ZERO_BASED_LABEL_INDEXING, true); // NOT STANDARD!
        config.setInt(SVMLightRecordReader.NUM_FEATURES, 11);
        config.setBoolean(SVMLightRecordReader.MULTILABEL, true);
        config.setInt(SVMLightRecordReader.NUM_LABELS, 5);
        rr.initialize(config, new FileSplit(new ClassPathResource("datavec-api/svmlight/multilabel.txt").getFile()));
        int i = 0;
        while (rr.hasNext()) {
            List<Writable> record = rr.next();
            assertEquals(correct.get(i), record);
            i++;
        }
        assertEquals(i, correct.size());
    }

    @Test
    public void testNextRecord() throws IOException, InterruptedException {
        SVMLightRecordReader rr = new SVMLightRecordReader();
        Configuration config = new Configuration();
        config.setBoolean(SVMLightRecordReader.ZERO_BASED_INDEXING, false);
        config.setInt(SVMLightRecordReader.NUM_FEATURES, 10);
        config.setBoolean(SVMLightRecordReader.APPEND_LABEL, false);
        rr.initialize(config, new FileSplit(new ClassPathResource("datavec-api/svmlight/basic.txt").getFile()));

        Record record = rr.nextRecord();
        List<Writable> recordList = record.getRecord();
        assertEquals(new DoubleWritable(1.0), recordList.get(1));
        assertEquals(new DoubleWritable(3.0), recordList.get(5));
        assertEquals(new DoubleWritable(4.0), recordList.get(7));

        record = rr.nextRecord();
        recordList = record.getRecord();
        assertEquals(new DoubleWritable(0.1), recordList.get(0));
        assertEquals(new DoubleWritable(6.6), recordList.get(5));
        assertEquals(new DoubleWritable(80.0), recordList.get(7));
    }

    @Test(expected = NoSuchElementException.class)
    public void testNoSuchElementException() throws Exception {
        SVMLightRecordReader rr = new SVMLightRecordReader();
        Configuration config = new Configuration();
        config.setInt(SVMLightRecordReader.NUM_FEATURES, 11);
        rr.initialize(config, new FileSplit(new ClassPathResource("datavec-api/svmlight/basic.txt").getFile()));
        while (rr.hasNext())
            rr.next();
        rr.next();
    }

    @Test(expected = UnsupportedOperationException.class)
    public void failedToSetNumFeaturesException() throws Exception {
        SVMLightRecordReader rr = new SVMLightRecordReader();
        Configuration config = new Configuration();
        rr.initialize(config, new FileSplit(new ClassPathResource("datavec-api/svmlight/basic.txt").getFile()));
        while (rr.hasNext())
            rr.next();
    }

    @Test(expected = UnsupportedOperationException.class)
    public void testInconsistentNumLabelsException() throws Exception {
        SVMLightRecordReader rr = new SVMLightRecordReader();
        Configuration config = new Configuration();
        config.setBoolean(SVMLightRecordReader.ZERO_BASED_INDEXING, false);
        rr.initialize(config, new FileSplit(new ClassPathResource("datavec-api/svmlight/inconsistentNumLabels.txt").getFile()));
        while (rr.hasNext())
            rr.next();
    }

    @Test(expected = UnsupportedOperationException.class)
    public void failedToSetNumMultiabelsException() throws Exception {
        SVMLightRecordReader rr = new SVMLightRecordReader();
        Configuration config = new Configuration();
        rr.initialize(config, new FileSplit(new ClassPathResource("datavec-api/svmlight/multilabel.txt").getFile()));
        while (rr.hasNext())
            rr.next();
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testFeatureIndexExceedsNumFeatures() throws Exception {
        SVMLightRecordReader rr = new SVMLightRecordReader();
        Configuration config = new Configuration();
        config.setInt(SVMLightRecordReader.NUM_FEATURES, 9);
        rr.initialize(config, new FileSplit(new ClassPathResource("datavec-api/svmlight/basic.txt").getFile()));
        rr.next();
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testLabelIndexExceedsNumLabels() throws Exception {
        SVMLightRecordReader rr = new SVMLightRecordReader();
        Configuration config = new Configuration();
        config.setInt(SVMLightRecordReader.NUM_FEATURES, 10);
        config.setInt(SVMLightRecordReader.NUM_LABELS, 6);
        rr.initialize(config, new FileSplit(new ClassPathResource("datavec-api/svmlight/basic.txt").getFile()));
        rr.next();
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testZeroIndexFeatureWithoutUsingZeroIndexing() throws Exception {
        SVMLightRecordReader rr = new SVMLightRecordReader();
        Configuration config = new Configuration();
        config.setBoolean(SVMLightRecordReader.ZERO_BASED_INDEXING, false);
        config.setInt(SVMLightRecordReader.NUM_FEATURES, 10);
        rr.initialize(config, new FileSplit(new ClassPathResource("datavec-api/svmlight/zeroIndexFeature.txt").getFile()));
        rr.next();
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testZeroIndexLabelWithoutUsingZeroIndexing() throws Exception {
        SVMLightRecordReader rr = new SVMLightRecordReader();
        Configuration config = new Configuration();
        config.setInt(SVMLightRecordReader.NUM_FEATURES, 10);
        config.setBoolean(SVMLightRecordReader.MULTILABEL, true);
        config.setInt(SVMLightRecordReader.NUM_LABELS, 2);
        rr.initialize(config, new FileSplit(new ClassPathResource("datavec-api/svmlight/zeroIndexLabel.txt").getFile()));
        rr.next();
    }
}
