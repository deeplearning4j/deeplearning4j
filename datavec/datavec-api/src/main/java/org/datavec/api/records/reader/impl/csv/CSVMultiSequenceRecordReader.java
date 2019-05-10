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

import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataInterval;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.writable.Writable;
import org.nd4j.base.Preconditions;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * CSVMultiSequenceRecordReader: Used to read CSV-format time series (sequence) data where there are multiple
 * independent sequences in each file.<br>
 * The assumption is that each sequence is separated by some delimiter - for example, a blank line between sequences,
 * or some other line that can be detected by a regex.<br>
 * Note that the number of columns (i.e., number of lines in the CSV per sequence) must be the same for all sequences.<br>
 * <br>
 * It supports 3 {@link Mode}s:<br>
 * (a) CONCAT mode: the output is a univariate (single column) sequence with the values from all lines
 * (b) EQUAL_LENGTH: Require that all lines have the exact same number of tokens<br>
 * (c) PAD: For any shorter lines (fewer tokens), a user-specified padding Writable value will be used to make them the same
 * length as the other sequences<br>
 * <br>
 * Example:<br>
 * Input data:
 * <pre>
 * {@code a,b,c
 *   1,2
 *
 *   A,B,C
 *   D,E,F}
 * </pre>
 * Output:<br>
 * (a) CONCAT: two sequences of length 5 and 6 respectively: [a,b,c,1,2] and [A,B,C,D,E,F]<br>
 * (b) EQUAL_LENGTH: Exception: because lines (a,b,c) and (1,2) have different lengths. If the second line was "1,2,3" instead,
 *    the output would be two sequences with 2 columns each, sequence length 3: [[a,b,c],[1,2,3]] and [[A,B,C],[D,E,F]]<br>
 * (c) PAD: two sequences with 2 columns each, sequence length 3: [[a,b,c],[1,2,PAD]] and [[A,B,C],[D,E,F]], where "PAD"
 *    is a user-specified padding value<br>
 * <br>
 * Note that the user has to specify a sequence separator regex: for "sequences are separated by an empty line" use "^$"
 *
 * @author Alex Black
 * @see CSVLineSequenceRecordReader CSVLineSequenceRecordReader for the edge case - a univariate version
 */
public class CSVMultiSequenceRecordReader extends CSVRecordReader implements SequenceRecordReader {

    public enum Mode {
        CONCAT,
        EQUAL_LENGTH,
        PAD
    }

    private String sequenceSeparatorRegex;
    private Mode mode;
    private Writable padValue;

    /**
     * Create a sequence reader using the default value for skip lines (0), the default delimiter (',') and the default
     * quote character ('"').<br>
     * Note that this constructor cannot be used with {@link Mode#PAD} as the padding value cannot be specified
     *
     * @param sequenceSeparatorRegex The sequence separator regex. Use "^$" for "sequences are separated by an empty line
     * @param mode                   Mode: see {@link CSVMultiSequenceRecordReader} javadoc
     */
    public CSVMultiSequenceRecordReader(String sequenceSeparatorRegex, Mode mode){
        this(0, DEFAULT_DELIMITER, DEFAULT_QUOTE, sequenceSeparatorRegex, mode, null);
    }

    /**
     * Create a sequence reader using the default value for skip lines (0), the default delimiter (',') and the default
     * quote character ('"')
     *
     * @param sequenceSeparatorRegex The sequence separator regex. Use "^$" for "sequences are separated by an empty line
     * @param mode                   Mode: see {@link CSVMultiSequenceRecordReader} javadoc
     * @param padValue               Padding value for padding short sequences. Only used/allowable with {@link Mode#PAD},
     *                               should be null otherwise
     */
    public CSVMultiSequenceRecordReader(String sequenceSeparatorRegex, Mode mode, Writable padValue){
        this(0, DEFAULT_DELIMITER, DEFAULT_QUOTE, sequenceSeparatorRegex, mode, padValue);
    }

    /**
     * Create a sequence reader using the default value for skip lines (0), the default delimiter (',') and the default
     * quote character ('"')
     *
     * @param skipNumLines           Number of lines to skip
     * @param elementDelimiter       Delimiter for elements - i.e., ',' if lines are comma separated
     * @param sequenceSeparatorRegex The sequence separator regex. Use "^$" for "sequences are separated by an empty line
     * @param mode                   Mode: see {@link CSVMultiSequenceRecordReader} javadoc
     * @param padValue               Padding value for padding short sequences. Only used/allowable with {@link Mode#PAD},
     *                               should be null otherwise
     */
    public CSVMultiSequenceRecordReader(int skipNumLines, char elementDelimiter, char quote, String sequenceSeparatorRegex,
                                        Mode mode, Writable padValue){
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
        List<String> lines = new ArrayList<>();
        try(BufferedReader br = new BufferedReader(new InputStreamReader(dataInputStream))){
            String line;
            while((line = br.readLine()) != null && !line.matches(sequenceSeparatorRegex)){
                lines.add(line);
            }
        }

        return parseLines(lines, uri, 0, lines.size());
    }

    @Override
    public SequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException {
        throw new UnsupportedOperationException("Not yet supported");
    }

    @Override
    public List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        throw new UnsupportedOperationException("Not yet supported");
    }

    @Override
    public boolean batchesSupported() {
        return false;
    }
}
