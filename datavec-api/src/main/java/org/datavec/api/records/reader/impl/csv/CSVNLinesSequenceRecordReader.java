/*
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

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.writable.Writable;

import java.io.*;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

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
    private int nLinesPerSequence;

    /**
     * @param nLinesPerSequence    Number of lines in each sequence
     */
    public CSVNLinesSequenceRecordReader(int nLinesPerSequence){
        this(nLinesPerSequence, 0, CSVRecordReader.DEFAULT_DELIMITER);
    }

    /**
     *
     * @param nLinesPerSequence    Number of lines in each sequences
     * @param skipNumLines         Number of lines to skip at the start of the file (only skipped once, not per sequence)
     * @param delimiter            Delimiter between entries in the same line, for example ","
     */
    public CSVNLinesSequenceRecordReader(int nLinesPerSequence, int skipNumLines, String delimiter){
        super(skipNumLines, delimiter);
        this.nLinesPerSequence = nLinesPerSequence;
    }

    @Override
    public List<List<Writable>> sequenceRecord() {
        if(!super.hasNext()){
            throw new NoSuchElementException("No next element");
        }

        List<List<Writable>> sequence = new ArrayList<>();
        int count = 0;
        while(count++ < nLinesPerSequence && super.hasNext()){
            sequence.add(super.next());
        }

        return sequence;
    }

    @Override
    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException("Reading CSV data from DataInputStream not yet implemented");
    }
}
