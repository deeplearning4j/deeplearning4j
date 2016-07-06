/*
 *
 *  *
 *  *  * Copyright 2015 Skymind,Inc.
 *  *  *
 *  *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *  *    you may not use this file except in compliance with the License.
 *  *  *    You may obtain a copy of the License at
 *  *  *
 *  *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *  *
 *  *  *    Unless required by applicable law or agreed to in writing, software
 *  *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *  *    See the License for the specific language governing permissions and
 *  *  *    limitations under the License.
 *  *
 *
 */

package org.canova.api.records.reader.impl;

import static org.junit.Assume.*;
import static org.junit.Assert.*;

import org.apache.commons.io.IOUtils;

import org.canova.api.conf.Configuration;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.writer.RecordWriter;
import org.canova.api.records.writer.impl.SVMLightRecordWriter;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.api.util.ClassPathResource;
import org.canova.api.writable.Writable;
import org.junit.Test;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 1/11/15.
 */
public class SVMRecordWriterTest {

    @Test
    public void testWriter() throws Exception {
        InputStream is  = new ClassPathResource("iris.dat").getInputStream();
        assumeNotNull(is);
        File tmp = new File("iris.txt");
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(tmp));
        IOUtils.copy(is, bos);
        bos.flush();
        bos.close();
        InputSplit split = new FileSplit(tmp);
        tmp.deleteOnExit();
        RecordReader reader = new CSVRecordReader();
        List<Collection<Writable>> records = new ArrayList<>();
        reader.initialize(split);
        while(reader.hasNext()) {
            Collection<Writable> record = reader.next();
            assertEquals(5, record.size());
            records.add(record);
        }

        assertEquals(150,records.size());
        File out = new File("iris_out.txt");
        out.deleteOnExit();
        RecordWriter writer = new SVMLightRecordWriter(out,true);
        for(Collection<Writable> record : records)
            writer.write(record);

        writer.close();
        records.clear();

        RecordReader svmReader = new SVMLightRecordReader();
        InputSplit svmSplit = new FileSplit(out);
        svmReader.initialize(svmSplit);
        assertTrue(svmReader.hasNext());
        while(svmReader.hasNext()) {
            Collection<Writable> record = svmReader.next();
            assertEquals(5, record.size());
            records.add(record);
        }
        assertEquals(150,records.size());
    }

    @Test
    public void testSparseData() throws Exception {
        RecordReader svmLightRecordReader = new SVMLightRecordReader();
        Configuration conf = new Configuration();
        conf.set(SVMLightRecordReader.NUM_ATTRIBUTES,"784");
        svmLightRecordReader.initialize(conf, new FileSplit(new ClassPathResource("mnist_svmlight.txt").getFile()));
        assertTrue(svmLightRecordReader.hasNext());
        Collection<Writable> record = svmLightRecordReader.next();
        assertEquals(785,record.size());
    }


}
