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

import org.canova.api.conf.Configuration;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.writer.RecordWriter;
import org.canova.api.records.writer.impl.FileRecordWriter;
import org.canova.api.records.writer.impl.LibSvmRecordWriter;
import org.canova.api.split.FileSplit;
import org.canova.api.util.ClassPathResource;
import org.canova.api.writable.Writable;
import org.junit.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
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
        File out = new File("iris.libsvm.out");
        if(out.exists()) out.delete();
        conf.set(FileRecordWriter.PATH, out.getAbsolutePath());
        RecordReader libSvmRecordReader = new LibSvmRecordReader();
        libSvmRecordReader.initialize(conf, new FileSplit(new ClassPathResource("iris.libsvm").getFile()));

        RecordWriter writer = new LibSvmRecordWriter();
        writer.setConf(conf);
        Collection<Collection<Writable>> data = new ArrayList<>();
        while (libSvmRecordReader.hasNext()) {
            Collection<Writable> record = libSvmRecordReader.next();
            writer.write(record);
            data.add(record);
        }
        writer.close();

        out.deleteOnExit();
        Collection<Collection<Writable>> test = new ArrayList<>();
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
        conf.set(LineRecordReader.APPEND_LABEL,"true");
        conf.set(LibSvmRecordReader.NUM_FEATURES, "4");
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
            for( int j=0; j<exp.size(); j++ ){
                assertEquals(exp.get(j),record.get(j).toDouble(),0.0);
            }
        }
    }


}
