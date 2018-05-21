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

package org.datavec.api.records.reader.impl.misc;


import org.datavec.api.records.reader.impl.FileRecordReader;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.StringReader;
import java.net.URI;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Matlab record reader
 *
 * @author Adam Gibson
 */
public class MatlabRecordReader extends FileRecordReader {

    private List<List<Writable>> records = new ArrayList<>();
    private Iterator<List<Writable>> currIter;

    @Override
    public boolean hasNext() {
        return super.hasNext();
    }

    @Override
    public List<Writable> next() {
        //use the current iterator
        if (currIter != null && currIter.hasNext())
            return new ArrayList<>(currIter.next());
        records.clear();
        //next file
        List<Writable> next = super.next();
        String val = next.iterator().next().toString();
        StringReader reader = new StringReader(val);
        int c;
        char chr;
        StringBuilder fileContent;
        boolean isComment;


        List<Writable> currRecord = new ArrayList<>();
        fileContent = new StringBuilder();
        isComment = false;
        records.add(currRecord);
        try {
            // determine number of attributes
            while ((c = reader.read()) != -1) {
                chr = (char) c;

                // comment found?
                if (chr == '%')
                    isComment = true;

                // end of line reached
                if ((chr == '\n') || (chr == '\r')) {
                    isComment = false;
                    if (fileContent.length() > 0)
                        currRecord.add(new DoubleWritable(new Double(fileContent.toString())));

                    if (currRecord.size() > 0) {
                        currRecord = new ArrayList<>();
                        records.add(currRecord);
                    }
                    fileContent = new StringBuilder();
                    continue;
                }

                // skip till end of comment line
                if (isComment)
                    continue;

                // separator found?
                if ((chr == '\t') || (chr == ' ')) {
                    if (fileContent.length() > 0) {
                        currRecord.add(new DoubleWritable(new Double(fileContent.toString())));
                        fileContent = new StringBuilder();
                    }
                } else {
                    fileContent.append(chr);
                }
            }

            // last number?
            if (fileContent.length() > 0)
                currRecord.add(new DoubleWritable(new Double(fileContent.toString())));


            currIter = records.iterator();

        } catch (Exception ex) {
            ex.printStackTrace();
            throw new IllegalStateException("Unable to determine structure as Matlab ASCII file: " + ex);
        }
        throw new IllegalStateException("Strange state detected");
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException("Reading Matlab data from DataInputStream: not yet implemented");
    }
}
