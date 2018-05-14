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
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVNLinesSequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.SequenceRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.FloatWritable;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.WritableType;
import org.datavec.hadoop.records.reader.mapfile.MapFileRecordReader;
import org.datavec.hadoop.records.reader.mapfile.MapFileSequenceRecordReader;
import org.datavec.hadoop.records.writer.mapfile.MapFileRecordWriter;
import org.datavec.hadoop.records.writer.mapfile.MapFileSequenceRecordWriter;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 07/07/2017.
 */
public class TestMapFileRecordWriter {

    @Test
    public void testWriter() throws Exception {

        for(boolean convertWritables : new boolean[]{false, true}) {

            File tempDirSingle = Files.createTempDir();
            File tempDirMultiple = Files.createTempDir();

            tempDirSingle.deleteOnExit();
            tempDirMultiple.deleteOnExit();

            WritableType textWritablesTo = convertWritables ? WritableType.Float : null;

            RecordWriter singlePartWriter = new MapFileRecordWriter(tempDirSingle, -1, textWritablesTo);
            RecordWriter multiPartWriter = new MapFileRecordWriter(tempDirMultiple, 30, textWritablesTo);

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

            List<List<Writable>> exp = new ArrayList<>();
            List<List<Writable>> s1 = new ArrayList<>();
            List<List<Writable>> s2 = new ArrayList<>();

            rr.reset();
            while (rr.hasNext()) {
                exp.add(rr.next());
            }

            while (rr1.hasNext()) {
                s1.add(rr1.next());
            }

            while (rr2.hasNext()) {
                s2.add(rr2.next());
            }

            assertEquals(150, exp.size());

            if(convertWritables){
                List<List<Writable>> asFloat = new ArrayList<>();
                for(List<Writable> l : exp ){
                    List<Writable> newList = new ArrayList<>();
                    for(Writable w : l){
                        newList.add(new FloatWritable(w.toFloat()));
                    }
                    asFloat.add(newList);
                }

                exp = asFloat;
            }

            assertEquals(exp, s1);
            assertEquals(exp, s2);


            //By default: we won't be doing any conversion of text types. CsvRecordReader outputs Text writables
            for (List<Writable> l : s1) {
                for (Writable w : l) {
                    if(convertWritables){
                        assertEquals(WritableType.Float, w.getType());
                    } else {
                        assertEquals(WritableType.Text, w.getType());
                    }
                }
            }
        }
    }


    @Test
    public void testSequenceWriter() throws Exception {

        for(boolean convertWritables : new boolean[]{false, true}) {

            File tempDirSingle = Files.createTempDir();
            File tempDirMultiple = Files.createTempDir();

            tempDirSingle.deleteOnExit();
            tempDirMultiple.deleteOnExit();

            WritableType textWritablesTo = convertWritables ? WritableType.Float : null;

            SequenceRecordWriter singlePartWriter = new MapFileSequenceRecordWriter(tempDirSingle, -1, textWritablesTo);
            SequenceRecordWriter multiPartWriter = new MapFileSequenceRecordWriter(tempDirMultiple, 10, textWritablesTo);

            SequenceRecordReader rr = new CSVNLinesSequenceRecordReader(5);
            ClassPathResource cpr = new ClassPathResource("iris.dat");
            rr.initialize(new FileSplit(cpr.getFile()));

            RecordReaderConverter.convert(rr, singlePartWriter);
            rr.reset();
            RecordReaderConverter.convert(rr, multiPartWriter);

            singlePartWriter.close();
            multiPartWriter.close();

            SequenceRecordReader rr1 = new MapFileSequenceRecordReader();
            SequenceRecordReader rr2 = new MapFileSequenceRecordReader();
            rr1.initialize(new FileSplit(tempDirSingle));
            rr2.initialize(new FileSplit(tempDirMultiple));

            List<List<List<Writable>>> exp = new ArrayList<>();
            List<List<List<Writable>>> s1 = new ArrayList<>();
            List<List<List<Writable>>> s2 = new ArrayList<>();

            rr.reset();
            while (rr.hasNext()) {
                exp.add(rr.sequenceRecord());
            }

            while (rr1.hasNext()) {
                s1.add(rr1.sequenceRecord());
            }

            while (rr2.hasNext()) {
                s2.add(rr2.sequenceRecord());
            }

            assertEquals(150/5, exp.size());

            if(convertWritables){
                List<List<List<Writable>>> asFloat = new ArrayList<>();
                for(List<List<Writable>> sequence : exp ){
                    List<List<Writable>> newSeq = new ArrayList<>();
                    for(List<Writable> step : sequence ){
                        List<Writable> newStep = new ArrayList<>();
                        for(Writable w : step){
                            newStep.add(new FloatWritable(w.toFloat()));
                        }
                        newSeq.add(newStep);
                    }
                    asFloat.add(newSeq);
                }
                exp = asFloat;
            }

            assertEquals(exp, s1);
            assertEquals(exp, s2);


            //By default: we won't be doing any conversion of text types. CsvRecordReader outputs Text writables
            for(List<List<Writable>> seq : s1) {
                for (List<Writable> l : seq) {
                    for (Writable w : l) {
                        if (convertWritables) {
                            assertEquals(WritableType.Float, w.getType());
                        } else {
                            assertEquals(WritableType.Text, w.getType());
                        }
                    }
                }
            }
        }
    }


}
