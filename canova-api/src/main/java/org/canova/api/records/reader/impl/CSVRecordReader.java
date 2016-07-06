/*
 *
 *  *
 *  *  * Copyright 2015 Skymind,Inc.
 *  *  *
 *  *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *  *    you may not use this file except in compliance with the License.
 *  *  *    You may obtain a copy of the License at
 *  *  *
 *  *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *  *
 *  *  *    Unless required by applicable law or agreed to in writing, software
 *  *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *  *    See the License for the specific language governing permissions and
 *  *  *    limitations under the License.
 *  *
 *
 */

package org.canova.api.records.reader.impl;



import org.canova.api.conf.Configuration;
import org.canova.api.io.data.Text;
import org.canova.api.split.InputSplit;
import org.canova.api.writable.Writable;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;

/**
 * Simple csv record reader.
 *
 * @author Adam Gibson
 */
public class CSVRecordReader extends LineRecordReader {
    private boolean skippedLines = false;
    private int skipNumLines = 0;
    private String delimiter = ",";
    public final static String SKIP_NUM_LINES = NAME_SPACE + ".skipnumlines";
    public final static String DELIMITER = NAME_SPACE + ".delimiter";

    /**
     * Skip first n lines
     * @param skipNumLines the number of lines to skip
     */
    public CSVRecordReader(int skipNumLines) {
         this(skipNumLines,",");
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
        this(0,",");
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        super.initialize(conf, split);
        this.skipNumLines = conf.getInt(SKIP_NUM_LINES,this.skipNumLines);
        this.delimiter = conf.get(DELIMITER, ","); 
    }

    @Override
    public Collection<Writable> next() {
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
    public Collection<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        //Here: we are reading a single line from the DataInputStream. How to handle skipLines???
        throw new UnsupportedOperationException("Reading CSV data from DataInputStream not yet implemented");
    }

    @Override
    public void reset() {
        super.reset();
        skippedLines = false;
    }
}
