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

package org.canova.api.records.writer.impl;


import org.canova.api.conf.Configuration;
import org.canova.api.writable.Writable;

import java.io.*;
import java.nio.charset.Charset;
import java.util.Collection;

/**
 * Csv record writer
 *
 * @author Adam Gibson
 */
public class CSVRecordWriter extends FileRecordWriter {
    public static final String DEFAULT_DELIMITER = ",";

    private final byte[] delimBytes;
    private boolean firstLine = true;

    public CSVRecordWriter() {
        delimBytes = DEFAULT_DELIMITER.getBytes(encoding);
    }

    public CSVRecordWriter(File path) throws FileNotFoundException {
        this(path,false,DEFAULT_CHARSET,DEFAULT_DELIMITER);
    }

    public CSVRecordWriter(File path, boolean append) throws FileNotFoundException {
        this(path,append,DEFAULT_CHARSET,DEFAULT_DELIMITER);
    }

    public CSVRecordWriter(Configuration conf) throws FileNotFoundException {
        super(conf);
        delimBytes = DEFAULT_DELIMITER.getBytes(encoding);
    }

    public CSVRecordWriter(File path, boolean append, Charset encoding, String delimiter) throws FileNotFoundException{
        super(path,append,encoding);
        this.delimBytes = delimiter.getBytes(encoding);
    }

    @Override
    public void write(Collection<Writable> record) throws IOException {
        if(!record.isEmpty()) {
            //Add new line before appending lines rather than after (avoids newline after last line)
            if(!firstLine){
                out.write(NEW_LINE.getBytes());
            } else {
                firstLine = false;
            }

            int count = 0;
            int last = record.size()-1;
            for(Writable w : record) {
                out.write(w.toString().getBytes(encoding));
                if(count++ != last) out.write(delimBytes);
            }

            out.flush();
        }
    }
}
