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

package org.datavec.hadoop.records.reader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import org.datavec.api.conf.Configuration;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.records.reader.impl.LineRecordReader;
import org.datavec.api.split.InputSplit;
// import org.datavec.api.records.reader.impl.misc.SVMLightRecordReader;
import org.datavec.api.writable.Writable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SVMLightRecordReader extends LineRecordReader {

    private static Logger log = LoggerFactory.getLogger(SVMLightRecordReader.class);

    public SVMLightRecordReader() {}

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        initialize(split);
    }

    @Override
    public boolean hasNext() {
        //    return iter != null && iter.hasNext();
        return false;
    }


    /**
     * next() method for getting another K/V pair off disk from the SVMLight text file
     * 
     */
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

            // read values into array
            tok = new StringTokenizer(val, " \t");

            // 1. class
            double classVal = Double.parseDouble(tok.nextToken());

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
                value = Double.parseDouble(col.substring(col.indexOf(":") + 1));
                ret.add(new DoubleWritable(value));
            }

            ret.add(new DoubleWritable(classVal));
        } catch (Exception e) {
            log.error("Error parsing line '" + val + "': ", e);
        }

        return ret;
    }


}
