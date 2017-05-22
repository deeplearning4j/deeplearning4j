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

import static org.junit.Assume.*;
import static org.junit.Assert.*;

import org.apache.commons.io.IOUtils;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.misc.SVMLightRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.impl.misc.SVMLightRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.datavec.api.conf.Configuration;
import org.junit.Test;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 1/11/15.
 */
public class SVMRecordWriterTest {

    @Test
    public void testWriter() throws Exception {
        String tempDir = System.getProperty("java.io.tmpdir");
        InputStream is = new ClassPathResource("iris.dat").getInputStream();
        assumeNotNull(is);
        File tmp = new File(tempDir, "iris.txt");
        if (tmp.exists())
            tmp.delete();
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(tmp));
        IOUtils.copy(is, bos);
        bos.flush();
        bos.close();
        InputSplit split = new FileSplit(tmp);
        tmp.deleteOnExit();
        RecordReader reader = new CSVRecordReader();
        List<List<Writable>> records = new ArrayList<>();
        reader.initialize(split);
        while (reader.hasNext()) {
            List<Writable> record = reader.next();
            assertEquals(5, record.size());
            records.add(record);
        }

        assertEquals(150, records.size());
        File out = new File(tempDir, "iris_out.txt");
        if (out.exists())
            out.delete();
        out.deleteOnExit();
        RecordWriter writer = new SVMLightRecordWriter(out, true);
        for (List<Writable> record : records)
            writer.write(record);

        writer.close();
        records.clear();

        Configuration conf = new Configuration();
        conf.setBoolean(SVMLightRecordReader.ZERO_BASED_INDEXING, false);
        RecordReader svmReader = new SVMLightRecordReader();
        InputSplit svmSplit = new FileSplit(out);
        svmReader.initialize(conf, svmSplit);
        assertTrue(svmReader.hasNext());
        while (svmReader.hasNext()) {
            List<Writable> record = svmReader.next();
            assertEquals(5, record.size());
            records.add(record);
        }
        assertEquals(150, records.size());
    }

    @Test
    public void testSparseData() throws Exception {
        RecordReader svmLightRecordReader = new SVMLightRecordReader();
        Configuration conf = new Configuration();
        conf.set(SVMLightRecordReader.NUM_ATTRIBUTES, "784");
        svmLightRecordReader.initialize(conf, new FileSplit(new ClassPathResource("mnist_svmlight.txt").getFile()));
        assertTrue(svmLightRecordReader.hasNext());
        List<Writable> record = svmLightRecordReader.next();
        assertEquals(785, record.size());
    }

    @Test
    public void testReadZeroIndexed() throws Exception {
        Configuration conf = new Configuration();
        conf.set(FileRecordReader.APPEND_LABEL, "true");
        RecordReader libSvmRecordReader = new SVMLightRecordReader();
        libSvmRecordReader.initialize(conf, new FileSplit(new ClassPathResource("iris.libsvm").getFile()));

        Configuration confZero = new Configuration();
        confZero.set(FileRecordReader.APPEND_LABEL, "true");
        confZero.setBoolean(SVMLightRecordReader.ZERO_BASED_INDEXING, true);
        RecordReader libSvmRecordReaderZero = new SVMLightRecordReader();
        libSvmRecordReaderZero.initialize(conf, new FileSplit(new ClassPathResource("iris_zero_indexed.libsvm").getFile()));

        Configuration confNoZero = new Configuration();
        confNoZero.set(FileRecordReader.APPEND_LABEL, "true");
        confNoZero.setBoolean(SVMLightRecordReader.ZERO_BASED_INDEXING, false);
        RecordReader libSvmRecordReaderNoZero = new SVMLightRecordReader();
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
