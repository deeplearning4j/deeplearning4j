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

package org.datavec.api.records.reader.impl.csv;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataLineInterval;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.primitives.Triple;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.*;

/**
 * A CSV Sequence record reader where:<br>
 * (a) all time series are in a single file<br>
 * (b) each time series is of the same length (specified in constructor)<br>
 * (c) no delimiter is used between time series<br>
 *
 * For example, with nLinesPerSequence=10, lines 0 to 9 are the first time series, 10 to 19 are the second, and so on.
 *
 * @author Alex Black
 */
public class CSVNLinesSequenceRecordReader extends CSVRecordReader implements SequenceRecordReader {

    public static final String LINES_PER_SEQUENCE = NAME_SPACE + ".nlinespersequence";

    private int nLinesPerSequence;
    private String delimiter;

    /**
     * No-arg constructor with the default number of lines per sequence (10)
     */
    public CSVNLinesSequenceRecordReader() {
        this(10);
    }

    /**
     * @param nLinesPerSequence    Number of lines in each sequence, use default delemiter(,) between entries in the same line
     */
    public CSVNLinesSequenceRecordReader(int nLinesPerSequence) {
        this(nLinesPerSequence, 0, String.valueOf(CSVRecordReader.DEFAULT_DELIMITER));
    }

    /**
     *
     * @param nLinesPerSequence    Number of lines in each sequences
     * @param skipNumLines         Number of lines to skip at the start of the file (only skipped once, not per sequence)
     * @param delimiter            Delimiter between entries in the same line, for example ","
     */
    public CSVNLinesSequenceRecordReader(int nLinesPerSequence, int skipNumLines, String delimiter) {
        super(skipNumLines);
        this.delimiter = delimiter;
        this.nLinesPerSequence = nLinesPerSequence;
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        super.initialize(conf, split);
        this.nLinesPerSequence = conf.getInt(LINES_PER_SEQUENCE, nLinesPerSequence);
    }

    @Override
    public List<List<Writable>> sequenceRecord() {
        if (!super.hasNext()) {
            throw new NoSuchElementException("No next element");
        }

        List<List<Writable>> sequence = new ArrayList<>();
        int count = 0;
        while (count++ < nLinesPerSequence && super.hasNext()) {
            sequence.add(super.next());
        }

        return sequence;
    }

    @Override
    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException("Reading CSV data from DataInputStream not yet implemented");
    }

    @Override
    public SequenceRecord nextSequence() {
        int lineBefore = lineIndex;
        List<List<Writable>> record = sequenceRecord();
        int lineAfter = lineIndex;
        URI uri = (locations == null || locations.length < 1 ? null : locations[splitIndex]);
        RecordMetaData meta = new RecordMetaDataLineInterval(lineBefore, lineAfter - 1, uri,
                        CSVNLinesSequenceRecordReader.class);
        return new org.datavec.api.records.impl.SequenceRecord(record, meta);
    }

    @Override
    public SequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return loadSequenceFromMetaData(Collections.singletonList(recordMetaData)).get(0);
    }

    @Override
    public List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        //First: create a sorted list of the RecordMetaData
        List<Triple<Integer, RecordMetaDataLineInterval, List<List<Writable>>>> list = new ArrayList<>();
        Iterator<RecordMetaData> iter = recordMetaDatas.iterator();
        int count = 0;
        while (iter.hasNext()) {
            RecordMetaData rmd = iter.next();
            if (!(rmd instanceof RecordMetaDataLineInterval)) {
                throw new IllegalArgumentException(
                                "Invalid metadata; expected RecordMetaDataLineInterval instance; got: " + rmd);
            }
            list.add(new Triple<>(count++, (RecordMetaDataLineInterval) rmd,
                            (List<List<Writable>>) new ArrayList<List<Writable>>()));
        }

        //Sort by starting line number:
        Collections.sort(list, new Comparator<Triple<Integer, RecordMetaDataLineInterval, List<List<Writable>>>>() {
            @Override
            public int compare(Triple<Integer, RecordMetaDataLineInterval, List<List<Writable>>> o1,
                            Triple<Integer, RecordMetaDataLineInterval, List<List<Writable>>> o2) {
                return Integer.compare(o1.getSecond().getLineNumberStart(), o2.getSecond().getLineNumberStart());
            }
        });

        Iterator<String> lineIter = getIterator(0); //TODO handle multi file case...
        int currentLineIdx = 0;
        String line = lineIter.next();
        while (currentLineIdx < skipNumLines) {
            line = lineIter.next();
            currentLineIdx++;
        }
        for (Triple<Integer, RecordMetaDataLineInterval, List<List<Writable>>> next : list) {
            int nextStartLine = next.getSecond().getLineNumberStart();
            int nextEndLine = next.getSecond().getLineNumberEnd();
            while (currentLineIdx < nextStartLine && lineIter.hasNext()) {
                line = lineIter.next();
                currentLineIdx++;
            }
            while (currentLineIdx <= nextEndLine && (lineIter.hasNext() || currentLineIdx == nextEndLine)) {
                String[] split = line.split(this.delimiter, -1);
                List<Writable> writables = new ArrayList<>();
                for (String s : split) {
                    writables.add(new Text(s));
                }
                next.getThird().add(writables);
                currentLineIdx++;
                if (lineIter.hasNext()) {
                    line = lineIter.next();
                }
            }
        }
        closeIfRequired(lineIter);

        //Now, sort by the original order:
        Collections.sort(list, new Comparator<Triple<Integer, RecordMetaDataLineInterval, List<List<Writable>>>>() {
            @Override
            public int compare(Triple<Integer, RecordMetaDataLineInterval, List<List<Writable>>> o1,
                            Triple<Integer, RecordMetaDataLineInterval, List<List<Writable>>> o2) {
                return Integer.compare(o1.getFirst(), o2.getFirst());
            }
        });

        //And return...
        List<SequenceRecord> out = new ArrayList<>();
        for (Triple<Integer, RecordMetaDataLineInterval, List<List<Writable>>> t : list) {
            out.add(new org.datavec.api.records.impl.SequenceRecord(t.getThird(), t.getSecond()));
        }

        return out;
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) {
        throw new UnsupportedOperationException("Not supported");
    }
}
