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
import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataInterval;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;
import org.nd4j.base.Preconditions;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.*;

public class CSVMultiSequenceRecordReader extends CSVRecordReader implements SequenceRecordReader {

    public enum Mode {
        CONCAT,
        EQUAL_LENGTH,
        PAD
    }

    private String sequenceSeparatorRegex;
    private Mode mode;
    private Writable padValue;

    public CSVMultiSequenceRecordReader(String sequenceSeparatorRegex, Mode mode){
        this(0, DEFAULT_DELIMITER, DEFAULT_QUOTE, sequenceSeparatorRegex, mode, null);
    }

    public CSVMultiSequenceRecordReader(String sequenceSeparatorRegex, Mode mode, Writable padValue){
        this(0, DEFAULT_DELIMITER, DEFAULT_QUOTE, sequenceSeparatorRegex, mode, padValue);
    }

    public CSVMultiSequenceRecordReader(int skipNumLines, char elementDelimiter, char quote, String sequenceSeparatorRegex, Mode mode, Writable padValue){
        super(skipNumLines, elementDelimiter, quote);
        Preconditions.checkState(mode != Mode.PAD || padValue != null, "Cannot use Mode.PAD with a null padding value. " +
                "Padding value must be passed to constructor ");
        this.sequenceSeparatorRegex = sequenceSeparatorRegex;
        this.mode = mode;
        this.padValue = padValue;
    }


    @Override
    public List<List<Writable>> sequenceRecord() {
        return nextSequence().getSequenceRecord();
    }

    @Override
    public SequenceRecord nextSequence() {
        if(!hasNext())
            throw new NoSuchElementException("No next element");

        List<String> lines = new ArrayList<>();
        int firstLine = lineIndex;
        int lastLine = lineIndex;
        while(super.hasNext()){
            String line = readStringLine();
            if(line.matches(sequenceSeparatorRegex)){
                lastLine = lineIndex;
                break;
            }
            lines.add(line);
        }

        //Process lines
        URI uri = (locations == null || locations.length < 1 ? null : locations[splitIndex]);
        List<List<Writable>> out = parseLines(lines, uri, firstLine, lastLine);


        return new org.datavec.api.records.impl.SequenceRecord(out, new RecordMetaDataInterval(firstLine, lastLine, uri));
    }

    private List<List<Writable>> parseLines(List<String> lines, URI uri, int firstLine, int lastLine){
        List<List<Writable>> out = new ArrayList<>();
        switch (mode){
            case CONCAT:
                //Output is univariate sequence - concat all lines
                for(String s : lines){
                    List<Writable> parsed = super.parseLine(s);
                    for(Writable w : parsed){
                        out.add(Collections.singletonList(w));
                    }
                }
                break;
            case EQUAL_LENGTH:
            case PAD:
                List<List<Writable>> columnWise = new ArrayList<>();
                int length = -1;
                int lineNum = 0;
                for(String s : lines) {
                    List<Writable> parsed = super.parseLine(s); //This is one COLUMN
                    columnWise.add(parsed);
                    lineNum++;
                    if(mode == Mode.PAD){
                        length = Math.max(length, parsed.size());
                    } else if(length < 0)
                        length = parsed.size();
                    else if(mode == Mode.EQUAL_LENGTH){
                        Preconditions.checkState(parsed.size() == length, "Invalid state: When using CSVMultiSequenceRecordReader, " +
                                "all lines (columns) must be the same length. Prior columns had " + length + " elements, line " +
                                lineNum + " in sequence has length " + parsed.size() + " (Sequence position: " + uri +
                                ", lines " + firstLine + " to " + lastLine + ")");
                    }
                }

                if(mode == Mode.PAD){
                    for(List<Writable> w : columnWise){
                        while(w.size() < length){
                            w.add(padValue);
                        }
                    }
                }

                //Transpose: from column-wise to row-wise
                for( int i=0; i<length; i++ ){
                    List<Writable> step = new ArrayList<>();
                    for( int j=0; j<columnWise.size(); j++ ){
                        step.add(columnWise.get(j).get(i));
                    }
                    out.add(step);
                }
                break;
        }
        return out;
    }

    @Override
    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        return null;
    }

    @Override
    public SequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return null;
    }

    @Override
    public List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        return null;
    }

    @Override
    public boolean batchesSupported() {
        return false;
    }

}
