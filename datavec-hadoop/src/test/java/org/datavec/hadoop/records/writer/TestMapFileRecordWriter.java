/*
 *  * Copyright 2017 Skymind, Inc.
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

package org.datavec.hadoop.records.writer;

import com.google.common.io.Files;
import org.datavec.api.records.converter.RecordReaderConverter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.datavec.hadoop.records.reader.mapfile.MapFileRecordReader;
import org.datavec.hadoop.records.writer.mapfile.MapFileRecordWriter;
import org.junit.Test;

import java.io.File;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Created by Alex on 07/07/2017.
 */
public class TestMapFileRecordWriter {

    @Test
    public void testWriter() throws Exception {

        File tempDirSingle = Files.createTempDir();
        File tempDirMultiple = Files.createTempDir();

        RecordWriter singlePartWriter = new MapFileRecordWriter(tempDirSingle);
        RecordWriter multiPartWriter = new MapFileRecordWriter(tempDirMultiple, 3, 10, null);

        RecordReader rr = new CSVRecordReader();
        ClassPathResource cpr = new ClassPathResource("iris.dat");
        rr.initialize(new FileSplit(cpr.getFile()));

        RecordReaderConverter.convert(rr, singlePartWriter);
        rr.reset();
        RecordReaderConverter.convert(rr, multiPartWriter);

        singlePartWriter.close();
        multiPartWriter.close();

        RecordReader rr1 = new MapFileRecordReader();
        RecordReader rr2 = new MapFileRecordReader();
        rr1.initialize(new FileSplit(tempDirSingle));
        rr2.initialize(new FileSplit(tempDirMultiple));

        Set<List<Writable>> exp = new HashSet<>();
        Set<List<Writable>> s1 = new HashSet<>();
        Set<List<Writable>> s2 = new HashSet<>();

        rr.reset();
        while (rr.hasNext()) {
            exp.add(rr.next());
        }

        while(rr1.hasNext()){
            s1.add(rr1.next());
        }

        while(rr2.hasNext()){
            s2.add(rr2.next());
        }

        assertEquals(150, exp.size());


        assertEquals(exp, s1);
        assertEquals(exp, s2);
    }


    @Test
    public void testTextConversion() {


        fail();
    }

}
