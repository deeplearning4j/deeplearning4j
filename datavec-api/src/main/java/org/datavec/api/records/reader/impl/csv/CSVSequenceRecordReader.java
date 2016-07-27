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

import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.impl.FileRecordReader;
import org.datavec.api.writable.Text;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.writable.Writable;

import java.io.*;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * CSV Sequence Record Reader
 * This reader is indended to read sequences of data in CSV format, where
 * each sequence is defined in its own file (and there are multiple files)
 * Each line in the file represents one time step
 * @author Alex Black
 */
public class CSVSequenceRecordReader extends FileRecordReader implements SequenceRecordReader {
    private int skipNumLines = 0;
    private String delimiter = ",";

    public CSVSequenceRecordReader() {
        this(0, ",");
    }

    public CSVSequenceRecordReader(int skipNumLines) {
        this(skipNumLines, ",");
    }

    public CSVSequenceRecordReader(int skipNumLines, String delimiter) {
        this.skipNumLines = skipNumLines;
        this.delimiter = delimiter;
    }

    @Override
    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        invokeListeners(uri);
        @SuppressWarnings("unchecked")
        Iterator<String> lineIter = IOUtils.lineIterator(new InputStreamReader(dataInputStream));
        if (skipNumLines > 0) {
            int count = 0;
            while (count++ < skipNumLines && lineIter.hasNext()) lineIter.next();
        }

        List<List<Writable>> out = new ArrayList<>();
        while (lineIter.hasNext()) {
            String line = lineIter.next();
            String[] split = line.split(delimiter);
            ArrayList<Writable> list = new ArrayList<>();
            for (String s : split) list.add(new Text(s));
            out.add(list);
        }

        return out;
    }

    @Override
    @SuppressWarnings("unchecked")
    public List<List<Writable>> sequenceRecord() {
        File next = iter.next();
        invokeListeners(next);

        Iterator<String> lineIter;
        try {
            lineIter = IOUtils.lineIterator(new InputStreamReader(new FileInputStream(next)));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        if (skipNumLines > 0) {
            int count = 0;
            while (count++ < skipNumLines && lineIter.hasNext()) lineIter.next();
        }

        List<List<Writable>> out = new ArrayList<>();
        while (lineIter.hasNext()) {
            String line = lineIter.next();
            String[] split = line.split(delimiter);
            ArrayList<Writable> list = new ArrayList<>();
            for (String s : split) list.add(new Text(s));
            out.add(list);
        }

        return out;
    }


}
