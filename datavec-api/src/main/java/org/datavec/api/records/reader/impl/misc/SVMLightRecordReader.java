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


import org.datavec.api.records.reader.impl.LineRecordReader;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;
import org.datavec.api.conf.Configuration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

/**
 * Adapted from the weka svmlight reader
 *
 *	June 2015
 *		-	adapted to understand HDFS-style block splits
 *
 * @author Adam Gibson
 * @author Josh Patterson
 */
public class SVMLightRecordReader extends LineRecordReader {
    private static Logger log = LoggerFactory.getLogger(SVMLightRecordReader.class);
    private int numAttributes = -1;
    public static final String NAME_SPACE = LibSvmRecordReader.class.getName();
    public final static String NUM_ATTRIBUTES = NAME_SPACE + ".numattributes";
    public static final String NUM_FEATURES = NAME_SPACE + ".numfeatures";
    public static final String ZERO_BASED_INDEXING = NAME_SPACE + ".zeroBasedIndexing";

    private boolean zeroBasedIndexing = true;

    public SVMLightRecordReader() {}

    @Override
    public List<Writable> next() {
        Text t = (Text) super.next().iterator().next();
        String val = new String(t.getBytes());
        List<Writable> ret = new ArrayList<>();
        StringTokenizer tok;
        int index, max;
        String col;
        double value;

        // actual data
        try {
            // determine max index
            max = 0;
            tok = new StringTokenizer(val, " \t");
            tok.nextToken(); // skip class
            while (tok.hasMoreTokens()) {
                col = tok.nextToken();
                // finished?
                if (col.startsWith("#"))
                    break;
                // qid is not supported
                if (col.startsWith("qid:"))
                    continue;
                // actual value
                index = Integer.parseInt(col.substring(0, col.indexOf(":")));
                if (index > max)
                    max = index;
            }

            if (numAttributes <= 0)
                numAttributes = max;

            if (max > numAttributes)
                throw new IndexOutOfBoundsException("Found " + max + " features in record, expected " + numAttributes);

            // read values into array
            tok = new StringTokenizer(val, " \t");

            // 1. class
            double classVal = Double.parseDouble(tok.nextToken());
            int numAttributesAdded = 0;
            // 2. attributes
            while (tok.hasMoreTokens()) {
                col = tok.nextToken();
                // finished?
                if (col.startsWith("#"))
                    break;
                // qid is not supported
                if (col.startsWith("qid:"))
                    continue;
                // actual value
                index = Integer.parseInt(col.substring(0, col.indexOf(":")));
                if (!zeroBasedIndexing)
                    index--;

                /* TODO: throw an exception here. */
                if (index < 0) throw new IndexOutOfBoundsException("Invalid data : found negative index");

                if (index > numAttributesAdded) {
                    int totalDiff = Math.abs(numAttributesAdded - index);
                    for (int i = numAttributesAdded; i < index; i++) {
                        ret.add(new DoubleWritable(0.0));

                    }
                    numAttributesAdded += totalDiff;
                }
                value = Double.parseDouble(col.substring(col.indexOf(":") + 1));
                ret.add(new DoubleWritable(value));
                numAttributesAdded++;
            }

            if (numAttributes >= 1 && ret.size() < numAttributes) {
                int totalDiff = Math.abs(ret.size() - numAttributes);
                for (int i = 0; i < totalDiff; i++) {
                    ret.add(new DoubleWritable(0.0));

                }
            }

            ret.add(new DoubleWritable(classVal));
        } catch (Exception e) {
            log.error("Error parsing line '" + val + "': ", e);
        }

        return ret;
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        super.initialize(conf, split);
        if (conf.get(NUM_ATTRIBUTES) != null)
            numAttributes = conf.getInt(NUM_ATTRIBUTES, -1);
        else if (conf.get(NUM_FEATURES) != null)
            numAttributes = conf.getInt(NUM_FEATURES, -1);
        zeroBasedIndexing = conf.getBoolean(ZERO_BASED_INDEXING, true);
    }

    @Override
    public void setConf(Configuration conf) {
        super.setConf(conf);
        if (conf.get(NUM_ATTRIBUTES) != null)
            numAttributes = conf.getInt(NUM_ATTRIBUTES, -1);
        else if (conf.get(NUM_FEATURES) != null)
            numAttributes = conf.getInt(NUM_FEATURES, -1);
        zeroBasedIndexing = conf.getBoolean(ZERO_BASED_INDEXING, true);
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        //Here: we are reading a single line from the DataInputStream. How to handle headers?
        throw new UnsupportedOperationException(
                        "Reading SVMLightRecordReader data from DataInputStream not yet implemented");
    }
}
