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
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.misc.LibSvmRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.impl.FileRecordWriter;
import org.datavec.api.records.writer.impl.misc.LibSvmRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.junit.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class LibSvmTest {
    @Test
    public void testReadWrite() throws Exception {
        Configuration conf = new Configuration();
        conf.set(FileRecordReader.APPEND_LABEL, "true");
        conf.setBoolean(LibSvmRecordReader.ZERO_BASED_INDEXING, false);
        File out = new File("iris.libsvm.out");
        if (out.exists())
            out.delete();
        conf.set(FileRecordWriter.PATH, out.getAbsolutePath());
        RecordReader libSvmRecordReader = new LibSvmRecordReader();
        libSvmRecordReader.initialize(conf, new FileSplit(new ClassPathResource("iris.libsvm").getFile()));

        RecordWriter writer = new LibSvmRecordWriter();
        writer.setConf(conf);
        List<List<Writable>> data = new ArrayList<>();
        while (libSvmRecordReader.hasNext()) {
            List<Writable> record = libSvmRecordReader.next();
            writer.write(record);
            data.add(record);
        }
        writer.close();

        out.deleteOnExit();
        List<List<Writable>> test = new ArrayList<>();
        RecordReader testLibSvmRecordReader = new LibSvmRecordReader();
        testLibSvmRecordReader.initialize(conf, new FileSplit(out));
        while (testLibSvmRecordReader.hasNext())
            test.add(testLibSvmRecordReader.next());
        assertEquals(data, test);
    }

    @Test
    public void testReadMissing() throws Exception {
        RecordReader rr = new LibSvmRecordReader();
        Configuration conf = new Configuration();
        conf.set(LineRecordReader.APPEND_LABEL, "true");
        conf.set(LibSvmRecordReader.NUM_FEATURES, "4");
        conf.setBoolean(LibSvmRecordReader.ZERO_BASED_INDEXING, false);
        rr.initialize(conf, new FileSplit(new ClassPathResource("libsvm_with_multiple_missing.libsvm").getFile()));

        List<List<Double>> expected = new ArrayList<>(11);
        expected.add(Arrays.asList(1.0, 2.0, 3.0, 4.0, 1.0));
        expected.add(Arrays.asList(0.0, 2.0, 3.0, 4.0, 2.0));
        expected.add(Arrays.asList(1.0, 0.0, 3.0, 4.0, 3.0));
        expected.add(Arrays.asList(1.0, 2.0, 0.0, 4.0, 4.0));
        expected.add(Arrays.asList(1.0, 2.0, 3.0, 0.0, 5.0));
        expected.add(Arrays.asList(0.0, 0.0, 3.0, 4.0, 6.0));
        expected.add(Arrays.asList(1.0, 0.0, 0.0, 4.0, 7.0));
        expected.add(Arrays.asList(1.0, 2.0, 0.0, 0.0, 8.0));
        expected.add(Arrays.asList(0.0, 0.0, 0.0, 4.0, 9.0));
        expected.add(Arrays.asList(1.0, 0.0, 0.0, 0.0, 10.0));
        expected.add(Arrays.asList(0.0, 0.0, 0.0, 0.0, 11.0));


        int count = 0;
        while (rr.hasNext()) {
            List<Writable> record = new ArrayList<>(rr.next());
            //            System.out.println(record);
            assertEquals(record.size(), 5);
            List<Double> exp = expected.get(count++);
            for (int j = 0; j < exp.size(); j++) {
                assertEquals(exp.get(j), record.get(j).toDouble(), 0.0);
            }
        }
    }

    @Test
    public void testReadZeroIndexed() throws Exception {
        Configuration conf = new Configuration();
        conf.set(FileRecordReader.APPEND_LABEL, "true");
        RecordReader libSvmRecordReader = new LibSvmRecordReader();
        libSvmRecordReader.initialize(conf, new FileSplit(new ClassPathResource("iris.libsvm").getFile()));

        Configuration confZero = new Configuration();
        confZero.set(FileRecordReader.APPEND_LABEL, "true");
        confZero.setBoolean(LibSvmRecordReader.ZERO_BASED_INDEXING, true);
        RecordReader libSvmRecordReaderZero = new LibSvmRecordReader();
        libSvmRecordReaderZero.initialize(conf, new FileSplit(new ClassPathResource("iris_zero_indexed.libsvm").getFile()));

        Configuration confNoZero = new Configuration();
        confNoZero.set(FileRecordReader.APPEND_LABEL, "true");
        confNoZero.setBoolean(LibSvmRecordReader.ZERO_BASED_INDEXING, false);
        RecordReader libSvmRecordReaderNoZero = new LibSvmRecordReader();
        libSvmRecordReaderNoZero.initialize(confNoZero, new FileSplit(new ClassPathResource("iris.libsvm").getFile()));

        while (libSvmRecordReader.hasNext()) {
            List<Writable> record = libSvmRecordReader.next();
            record.remove(0);
            List<Writable> recordZero = libSvmRecordReaderZero.next();
            assertEquals(record, recordZero);
            List<Writable> recordNoZero = libSvmRecordReaderNoZero.next();
            assertEquals(record, recordNoZero);
        }
    }
}
