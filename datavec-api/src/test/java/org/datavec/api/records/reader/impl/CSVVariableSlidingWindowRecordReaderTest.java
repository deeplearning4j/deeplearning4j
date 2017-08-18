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
import org.datavec.api.records.reader.impl.csv.CSVVariableSlidingWindowRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Tests for variable sliding window csv reader.
 *
 * @author Justin Long (crockpotveggies)
 */
public class CSVVariableSlidingWindowRecordReaderTest {

    @Test
    public void testCSVVariableSlidingWindowRecordReader() throws Exception {
        int maxLinesPerSequence = 3;

        SequenceRecordReader seqRR = new CSVVariableSlidingWindowRecordReader(maxLinesPerSequence, 1);
        seqRR.initialize(new FileSplit(new ClassPathResource("iris.dat").getFile()));

        CSVRecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource("iris.dat").getFile()));

        int count = 0;
        while (seqRR.hasNext()) {
            List<List<Writable>> next = seqRR.sequenceRecord();

            if(count==maxLinesPerSequence-1) {
                List<List<Writable>> expected = new ArrayList<>();
                for (int i = 0; i < maxLinesPerSequence; i++) {
                    expected.add(rr.next());
                }
                Collections.reverse(expected);
                assertEquals(expected, next);

            }
            if(count==maxLinesPerSequence) {
                assertEquals(maxLinesPerSequence, next.size());
            }
            if(count==0) { // first seq should be length 1
                assertEquals(1, next.size());
            }
            if(count==151) { // last seq should be length 1
                assertEquals(1, next.size());
            }

            count++;
        }
    }

}
