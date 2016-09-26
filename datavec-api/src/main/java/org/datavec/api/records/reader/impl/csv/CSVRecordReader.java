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



import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataLine;
import org.datavec.api.records.reader.RecordReaderMeta;
import org.datavec.api.records.reader.impl.LineRecordReader;
import org.datavec.api.writable.Text;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.*;

/**
 * Simple csv record reader.
 *
 * @author Adam Gibson
 */
public class CSVRecordReader extends LineRecordReader implements RecordReaderMeta {
    /** A regex delimiter that can parse quotes (string literals) that may have commas in them: http://stackoverflow.com/a/1757107/523744
     * Note: This adds considerable overhead compared to the default "," delimiter, and should only be used when necessary.
     * */
    public final static String QUOTE_HANDLING_DELIMITER = ",(?=([^\"]*\"[^\"]*\")*[^\"]*$)";
    private boolean skippedLines = false;
    private int skipNumLines = 0;
    private String delimiter = DEFAULT_DELIMITER;
    public final static String DEFAULT_DELIMITER = ",";
    public final static String SKIP_NUM_LINES = NAME_SPACE + ".skipnumlines";
    public final static String DELIMITER = NAME_SPACE + ".delimiter";

    /**
     * Skip first n lines
     * @param skipNumLines the number of lines to skip
     */
    public CSVRecordReader(int skipNumLines) {
        this(skipNumLines,DEFAULT_DELIMITER);
    }

    /**
     * Skip lines and use delimiter
     * @param skipNumLines the number of lines to skip
     * @param delimiter the delimiter
     */
    public CSVRecordReader(int skipNumLines,String delimiter) {
        this.skipNumLines = skipNumLines;
        this.delimiter = delimiter;
    }

    public CSVRecordReader() {
        this(0,DEFAULT_DELIMITER);
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        super.initialize(conf, split);
        this.skipNumLines = conf.getInt(SKIP_NUM_LINES,this.skipNumLines);
        this.delimiter = conf.get(DELIMITER, DEFAULT_DELIMITER);
    }

    @Override
    public List<Writable> next() {
        if(!skippedLines && skipNumLines > 0) {
            for(int i = 0; i < skipNumLines; i++) {
                if(!hasNext()) {
                    return new ArrayList<>();
                }
                super.next();
            }
            skippedLines = true;
        }
        Text t =  (Text) super.next().iterator().next();
        String val = t.toString();
        String[] split = val.split(delimiter, -1);
        List<Writable> ret = new ArrayList<>();
        for(String s : split)
            ret.add(new Text(s));
        return ret;
    }

    @Override
    public Record nextRecord(){
        List<Writable> next = next();
        URI uri = (locations == null || locations.length < 1 ? null : locations[0]);    //TODO: handle String splits etc
        RecordMetaData meta = new RecordMetaDataLine(this.currIndex-1, uri, CSVRecordReader.class); //-1 as line number has been incremented already...
        return new Record(next, meta);
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return loadFromMetaData(Collections.singletonList(recordMetaData)).get(0);
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        List<Record> list = super.loadFromMeta(recordMetaDatas);

        for(Record r : list ){
            String line = r.getRecord().get(0).toString();
            String[] split = line.split(delimiter, -1);
            List<Writable> writables = new ArrayList<>();
            for(String s : split) {
                writables.add(new Text(s));
            }
            r.setRecord(writables);
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
