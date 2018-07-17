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
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataLine;
import org.datavec.api.records.reader.impl.LineRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Simple csv record reader.
 *
 * @author Adam Gibson
 */
public class CSVRecordReader extends LineRecordReader {
    private boolean skippedLines = false;
    protected int skipNumLines = 0;
    public final static char DEFAULT_DELIMITER = ',';
    public final static char DEFAULT_QUOTE = '\"';
    public final static String SKIP_NUM_LINES = NAME_SPACE + ".skipnumlines";
    public final static String DELIMITER = NAME_SPACE + ".delimiter";
    public final static String QUOTE = NAME_SPACE + ".quote";

    private SerializableCSVParser csvParser;

    /**
     * Skip first n lines
     * @param skipNumLines the number of lines to skip
     */
    public CSVRecordReader(int skipNumLines) {
        this(skipNumLines, DEFAULT_DELIMITER);
    }

    /**
     * Create a CSVRecordReader with the specified delimiter
     * @param delimiter Delimiter character for CSV
     */
    public CSVRecordReader(char delimiter){
        this(0, delimiter);
    }

    /**
     * Skip lines and use delimiter
     * @param skipNumLines the number of lines to skip
     * @param delimiter the delimiter
     */
    public CSVRecordReader(int skipNumLines, char delimiter) {
        this(skipNumLines, delimiter, '\"');
    }

    /**
     *
     * @param skipNumLines Number of lines to skip
     * @param delimiter    Delimiter to use
     * @deprecated This constructor is deprecated; use {@link #CSVRecordReader(int, char)} or
     * {@link #CSVRecordReader(int, char, char)}
     */
    @Deprecated
    public CSVRecordReader(int skipNumLines, String delimiter){
        this(skipNumLines, stringDelimToChar(delimiter));
    }

    private static char stringDelimToChar(String delimiter) {
        if(delimiter.length() > 1){
            throw new UnsupportedOperationException("Multi-character delimiters have been deprecated. For quotes, " +
                    "use CSVRecordReader(int skipNumLines, char delimiter, char quote)");
        }
        return delimiter.charAt(0);
    }

    /**
     * Skip lines, use delimiter, and strip quotes
     * @param skipNumLines the number of lines to skip
     * @param delimiter the delimiter
     * @param quote the quote to strip
     */
    public CSVRecordReader(int skipNumLines, char delimiter, char quote) {
        this.skipNumLines = skipNumLines;
        this.csvParser = new SerializableCSVParser(delimiter, quote);
    }

    /**
     * Skip lines, use delimiter, and strip quotes
     * @param skipNumLines the number of lines to skip
     * @param delimiter the delimiter
     * @param quote the quote to strip
     * @deprecated This constructor is deprecated; use {@link #CSVRecordReader(int, char)} or
     * {@link #CSVRecordReader(int, char, char)}
     */
    @Deprecated
    public CSVRecordReader(int skipNumLines, String delimiter, String quote) {
        this(skipNumLines, stringDelimToChar(delimiter), stringDelimToChar(quote));
    }

    public CSVRecordReader() {
        this(0, DEFAULT_DELIMITER);
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        super.initialize(conf, split);
        this.skipNumLines = conf.getInt(SKIP_NUM_LINES, this.skipNumLines);
        this.csvParser = new SerializableCSVParser(conf.getChar(DELIMITER, DEFAULT_DELIMITER), conf.getChar(QUOTE, DEFAULT_QUOTE));
    }

    private boolean skipLines() {
        if (!skippedLines && skipNumLines > 0) {
            for (int i = 0; i < skipNumLines; i++) {
                if (!super.hasNext()) {
                    return false;
                }
                super.next();
            }
            skippedLines = true;
        }
        return true;
    }

    @Override
    public boolean batchesSupported() {
        return true;
    }

    @Override
    public boolean hasNext() {
        return skipLines() && super.hasNext();
    }

    @Override
    public List<List<Writable>> next(int num) {
        List<List<Writable>> ret = new ArrayList<>(num);
        int recordsRead = 0;
        while(hasNext() && recordsRead++ < num) {
            ret.add(next());
        }

        return ret;
    }

    @Override
    public List<Writable> next() {
        if (!skipLines())
            throw new NoSuchElementException("No next element found!");
        Text t = (Text) super.next().iterator().next();
        String val = t.toString();
        return parseLine(val);
    }

    protected List<Writable> parseLine(String line) {
        String[] split;
        try {
            split = csvParser.parseLine(line);
        } catch(IOException e) {
            throw new RuntimeException(e);
        }
        List<Writable> ret = new ArrayList<>();
        for (String s : split) {
            ret.add(new Text(s));
        }
        return ret;
    }

    @Override
    public Record nextRecord() {
        List<Writable> next = next();
        URI uri = (locations == null || locations.length < 1 ? null : locations[splitIndex]);
        RecordMetaData meta = new RecordMetaDataLine(this.lineIndex - 1, uri, CSVRecordReader.class); //-1 as line number has been incremented already...
        return new org.datavec.api.records.impl.Record(next, meta);
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return loadFromMetaData(Collections.singletonList(recordMetaData)).get(0);
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        List<Record> list = super.loadFromMetaData(recordMetaDatas);

        for (Record r : list) {
            String line = r.getRecord().get(0).toString();
            r.setRecord(parseLine(line));
        }

        return list;
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        //Here: we are reading a single line from the DataInputStream. How to handle skipLines???
        throw new UnsupportedOperationException("Reading CSV data from DataInputStream not yet implemented");
    }

    @Override
    public void reset() {
        super.reset();
        skippedLines = false;
    }

    @Override
    protected void onLocationOpen(URI location) {
        skippedLines = false;
    }
}
