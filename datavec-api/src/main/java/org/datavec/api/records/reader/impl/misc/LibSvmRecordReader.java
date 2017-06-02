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

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.conf.Configuration;
import org.datavec.api.exceptions.DataVecException;
import org.datavec.api.records.reader.impl.LineRecordReader;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Adam Gibson
 */
@Slf4j
public class LibSvmRecordReader extends LineRecordReader {

    public static final String CLASSIFICATION = "libsvm.classification";
    public static final String NAME_SPACE = LibSvmRecordReader.class.getName();
    public static final String NUM_FEATURES = NAME_SPACE + ".numfeatures";
    public static final String ZERO_BASED_INDEXING = NAME_SPACE + ".zeroBasedIndexing";

    private boolean appendLabel = false;
    private boolean classification = true;
    private int numFeatures;
    private boolean zeroBasedIndexing = true;

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        super.initialize(split);
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        super.initialize(conf, split);
        appendLabel = conf.getBoolean(APPEND_LABEL, false);
        classification = conf.getBoolean(CLASSIFICATION, true);
        numFeatures = conf.getInt(NUM_FEATURES, 0);
        zeroBasedIndexing = conf.getBoolean(ZERO_BASED_INDEXING, true);
    }

    @Override
    public List<Writable> next() {
        Text record2 = (Text) super.next().iterator().next();
        String line = record2.toString();


        String[] tokens = line.trim().split("\\s+");
        Double response;
        try {
            response = Integer.valueOf(tokens[0]).doubleValue();
        } catch (NumberFormatException e) {
            try {
                response = Double.valueOf(tokens[0]);
                classification = false;
            } catch (NumberFormatException ex) {
                System.err.println(ex);
                throw new NumberFormatException("Unrecognized response variable value: " + tokens[0]);
            }
        }


        tokens = line.trim().split("\\s+");

        List<Writable> record = new ArrayList<>();
        int read = 0;
        for (int k = 1; k < tokens.length; k++) {
            String[] pair = tokens[k].split(":");
            if (pair.length != 2) {
                throw new NumberFormatException("Invalid data: " + tokens[k]);
            }

            int j = Integer.valueOf(pair[0]);
            if (!zeroBasedIndexing)
                j = j - 1;

            if (j < 0) throw new IndexOutOfBoundsException("Invalid data, negative index.");

            while (j != read) {
                record.add(new DoubleWritable(0.0));
                read++;
            }
            try {
                int x = Integer.valueOf(pair[1]);
                record.add(new IntWritable(x));
            } catch (NumberFormatException e) {
                double x = Double.valueOf(pair[1]);
                record.add(new DoubleWritable(x));
            }
            read++;
        }
        while (read < numFeatures) {
            record.add(new DoubleWritable(0.0));
            read++;
        }
        if (numFeatures == 0)
            numFeatures = read;

        if (read > numFeatures)
            throw new IndexOutOfBoundsException("Found " + read + " features in record, expected " + numFeatures);

        if (classification && appendLabel || !classification) {
            record.add(new DoubleWritable(response));
        }

        return record;
    }

    @Override
    public boolean hasNext() {
        return super.hasNext();
    }

    @Override
    public void close() throws IOException {
        super.close();
    }

    @Override
    public void setConf(Configuration conf) {
        super.setConf(conf);
    }

    @Override
    public Configuration getConf() {
        return super.getConf();
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        //Here: we are reading a single line from the DataInputStream. How to handle headers?
        throw new UnsupportedOperationException("Reading LibSVM data from DataInputStream not yet implemented");
    }

}
