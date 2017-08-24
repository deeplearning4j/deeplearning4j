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
 * A sliding window of variable size across an entire CSV.
 *
 * In practice the sliding window size starts at 1, then linearly increase to maxLinesPer sequence, then
 * linearly decrease back to 1.
 *
 * @author Justin Long (crockpotveggies)
 */
public class CSVVariableSlidingWindowRecordReader extends CSVRecordReader implements SequenceRecordReader {

    public static final String LINES_PER_SEQUENCE = NAME_SPACE + ".nlinespersequence";

    private int maxLinesPerSequence;
    private String delimiter;
    private int stride;
    private LinkedList<List<Writable>> queue;
    private boolean exhausted;

    /**
     * No-arg constructor with the default number of lines per sequence (10)
     */
    public CSVVariableSlidingWindowRecordReader() {
        this(10, 1);
    }

    /**
     * @param maxLinesPerSequence Number of lines in each sequence, use default delemiter(,) between entries in the same line
     */
    public CSVVariableSlidingWindowRecordReader(int maxLinesPerSequence) {
        this(maxLinesPerSequence, 0, 1, String.valueOf(CSVRecordReader.DEFAULT_DELIMITER));
    }

    /**
     * @param maxLinesPerSequence Number of lines in each sequence, use default delemiter(,) between entries in the same line
     * @param stride Number of lines between records (increment window > 1 line)
     */
    public CSVVariableSlidingWindowRecordReader(int maxLinesPerSequence, int stride) {
        this(maxLinesPerSequence, 0, stride, String.valueOf(CSVRecordReader.DEFAULT_DELIMITER));
    }

    /**
     * @param maxLinesPerSequence Number of lines in each sequence, use default delemiter(,) between entries in the same line
     * @param stride Number of lines between records (increment window > 1 line)
     */
    public CSVVariableSlidingWindowRecordReader(int maxLinesPerSequence, int stride, String delimiter) {
        this(maxLinesPerSequence, 0, stride, String.valueOf(CSVRecordReader.DEFAULT_DELIMITER));
    }

    /**
     *
     * @param maxLinesPerSequence Number of lines in each sequences
     * @param skipNumLines Number of lines to skip at the start of the file (only skipped once, not per sequence)
     * @param stride Number of lines between records (increment window > 1 line)
     * @param delimiter Delimiter between entries in the same line, for example ","
     */
    public CSVVariableSlidingWindowRecordReader(int maxLinesPerSequence, int skipNumLines, int stride, String delimiter) {
        super(skipNumLines);
        if(stride < 1)
            throw new IllegalArgumentException("Stride must be greater than 1");

        this.delimiter = delimiter;
        this.maxLinesPerSequence = maxLinesPerSequence;
        this.stride = stride;
        this.queue = new LinkedList<>();
        this.exhausted = false;
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        super.initialize(conf, split);
        this.maxLinesPerSequence = conf.getInt(LINES_PER_SEQUENCE, maxLinesPerSequence);
    }

    @Override
    public boolean hasNext() {
        boolean moreInCsv = super.hasNext();
        boolean moreInQueue = !queue.isEmpty();
        return moreInCsv || moreInQueue;
    }

    @Override
    public List<List<Writable>> sequenceRecord() {
        // try polling next(), otherwise empty the queue
        // loop according to stride size
        for(int i = 0; i < stride; i++) {
            if(super.hasNext())
                queue.addFirst(super.next());
            else
                exhausted = true;

            if (exhausted && queue.size() < 1)
                throw new NoSuchElementException("No next element");

            if (queue.size() > maxLinesPerSequence || exhausted)
                queue.pollLast();
        }

        List<List<Writable>> sequence = new ArrayList<>();
        for(List<Writable> line : queue) {
            sequence.add(line);
        }

        if(exhausted && queue.size()==1)
            queue.pollLast();

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
        int lineAfter = lineIndex + queue.size();
        URI uri = (locations == null || locations.length < 1 ? null : locations[splitIndex]);
        RecordMetaData meta = new RecordMetaDataLineInterval(lineBefore, lineAfter - 1, uri,
                        CSVVariableSlidingWindowRecordReader.class);
        return new org.datavec.api.records.impl.SequenceRecord(record, meta);
    }

    @Override
    public SequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return loadSequenceFromMetaData(Collections.singletonList(recordMetaData)).get(0);
    }

    @Override
    public List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void reset() {
        super.reset();
        queue = new LinkedList<>();
        exhausted = false;
    }
}
