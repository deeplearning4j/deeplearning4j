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

import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVNLinesSequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 19/09/2016.
 */
public class CSVNLinesSequenceRecordReaderTest {

    @Test
    public void testCSVNLinesSequenceRecordReader() throws Exception {
        int nLinesPerSequence = 10;

        SequenceRecordReader seqRR = new CSVNLinesSequenceRecordReader(nLinesPerSequence);
        seqRR.initialize(new FileSplit(new ClassPathResource("iris.dat").getFile()));

        CSVRecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource("iris.dat").getFile()));

        int count = 0;
        while (seqRR.hasNext()) {
            List<List<Writable>> next = seqRR.sequenceRecord();

            List<List<Writable>> expected = new ArrayList<>();
            for (int i = 0; i < nLinesPerSequence; i++) {
                expected.add(rr.next());
            }

            assertEquals(10, next.size());
            assertEquals(expected, next);

            count++;
        }

        assertEquals(150 / nLinesPerSequence, count);
    }

    @Test
    public void testCSVNlinesSequenceRecordReaderMetaData() throws Exception {
        int nLinesPerSequence = 10;

        SequenceRecordReader seqRR = new CSVNLinesSequenceRecordReader(nLinesPerSequence);
        seqRR.initialize(new FileSplit(new ClassPathResource("iris.dat").getFile()));

        CSVRecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource("iris.dat").getFile()));

        List<List<List<Writable>>> out = new ArrayList<>();
        while (seqRR.hasNext()) {
            List<List<Writable>> next = seqRR.sequenceRecord();
            out.add(next);
        }

        seqRR.reset();
        List<List<List<Writable>>> out2 = new ArrayList<>();
        List<SequenceRecord> out3 = new ArrayList<>();
        List<RecordMetaData> meta = new ArrayList<>();
        while (seqRR.hasNext()) {
            SequenceRecord seq = seqRR.nextSequence();
            out2.add(seq.getSequenceRecord());
            meta.add(seq.getMetaData());
            out3.add(seq);
        }

        assertEquals(out, out2);

        List<SequenceRecord> out4 = seqRR.loadSequenceFromMetaData(meta);
        assertEquals(out3, out4);
    }

}
