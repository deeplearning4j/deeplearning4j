/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.api.records.reader.impl.csv;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataLineInterval;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.*;

public class CSVVariableSlidingWindowRecordReader extends CSVRecordReader implements SequenceRecordReader {

    public static final String LINES_PER_SEQUENCE = NAME_SPACE + ".nlinespersequence";

    private int maxLinesPerSequence;
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
        this(maxLinesPerSequence, 0, 1, CSVRecordReader.DEFAULT_DELIMITER);
    }

    /**
     * @param maxLinesPerSequence Number of lines in each sequence, use default delemiter(,) between entries in the same line
     * @param stride Number of lines between records (increment window > 1 line)
     */
    public CSVVariableSlidingWindowRecordReader(int maxLinesPerSequence, int stride) {
        this(maxLinesPerSequence, 0, stride, CSVRecordReader.DEFAULT_DELIMITER);
    }

    /**
     * @param maxLinesPerSequence Number of lines in each sequence, use default delemiter(,) between entries in the same line
     * @param stride Number of lines between records (increment window > 1 line)
     * @deprecated Use the constructor using char for delimiter instead
     */
    @Deprecated
    public CSVVariableSlidingWindowRecordReader(int maxLinesPerSequence, int stride, String delimiter) {
        this(maxLinesPerSequence, 0, stride, CSVRecordReader.DEFAULT_DELIMITER);
    }

    /**
     *
     * @param maxLinesPerSequence Number of lines in each sequences
     * @param skipNumLines Number of lines to skip at the start of the file (only skipped once, not per sequence)
     * @param stride Number of lines between records (increment window > 1 line)
     * @param delimiter Delimiter between entries in the same line, for example ","
     * @deprecated Use the constructor using char for delimiter instead
     */
    @Deprecated
    public CSVVariableSlidingWindowRecordReader(int maxLinesPerSequence, int skipNumLines, int stride, String delimiter) {
        super(skipNumLines, delimiter.charAt(0));
        if(stride < 1)
            throw new IllegalArgumentException("Stride must be greater than 1");

        this.maxLinesPerSequence = maxLinesPerSequence;
        this.stride = stride;
        this.queue = new LinkedList<>();
        this.exhausted = false;
    }

    /**
     * @param maxLinesPerSequence Number of lines in each sequence, use default delemiter(,) between entries in the same line
     * @param stride Number of lines between records (increment window > 1 line)
     */
    public CSVVariableSlidingWindowRecordReader(int maxLinesPerSequence, int stride, char delimiter) {
        this(maxLinesPerSequence, 0, stride, delimiter);
    }

    /**
     *
     * @param maxLinesPerSequence Number of lines in each sequences
     * @param skipNumLines Number of lines to skip at the start of the file (only skipped once, not per sequence)
     * @param stride Number of lines between records (increment window > 1 line)
     * @param delimiter Delimiter between entries in the same line, for example ","
     */
    public CSVVariableSlidingWindowRecordReader(int maxLinesPerSequence, int skipNumLines, int stride, char delimiter) {
        super(skipNumLines, delimiter);
        if(stride < 1)
            throw new IllegalArgumentException("Stride must be greater than 1");

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
        sequence.addAll(queue);

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
