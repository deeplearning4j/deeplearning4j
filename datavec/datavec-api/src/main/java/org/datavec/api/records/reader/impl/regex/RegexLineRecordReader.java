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

package org.datavec.api.records.reader.impl.regex;

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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * RegexLineRecordReader: Read a file, one line at a time, and split it into fields using a regex.
 * Specifically, we are using {@link java.util.regex.Pattern} and {@link java.util.regex.Matcher}.<br>
 * To load an entire file using a
 *
 * Example: Data in format "2016-01-01 23:59:59.001 1 DEBUG First entry message!"<br>
 * using regex String "(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\.\\d{3}) (\\d+) ([A-Z]+) (.*)"<br>
 * would be split into 4 Text writables: ["2016-01-01 23:59:59.001", "1", "DEBUG", "First entry message!"]
 *
 * @author Alex Black
 */
public class RegexLineRecordReader extends LineRecordReader {
    public final static String SKIP_NUM_LINES = NAME_SPACE + ".skipnumlines";

    private String regex;
    private int skipNumLines;
    private Pattern pattern;
    private int numLinesSkipped;
    private int currLine = 0;

    public RegexLineRecordReader(String regex, int skipNumLines) {
        this.regex = regex;
        this.skipNumLines = skipNumLines;
        this.pattern = Pattern.compile(regex);
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        super.initialize(conf, split);
        this.skipNumLines = conf.getInt(SKIP_NUM_LINES, this.skipNumLines);
    }

    @Override
    public List<Writable> next() {
        if (numLinesSkipped < skipNumLines) {
            for (int i = numLinesSkipped; i < skipNumLines; i++, numLinesSkipped++) {
                if (!hasNext()) {
                    return new ArrayList<>();
                }
                super.next();
            }
        }
        Text t = (Text) super.next().iterator().next();
        String val = t.toString();
        return parseLine(val);
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        Writable w = super.record(uri, dataInputStream).get(0);
        return parseLine(w.toString());
    }

    private List<Writable> parseLine(String line) {
        Matcher m = pattern.matcher(line);

        List<Writable> ret;
        if (m.matches()) {
            int count = m.groupCount();
            ret = new ArrayList<>(count);
            for (int i = 1; i <= count; i++) { //Note: Matcher.group(0) is the entire sequence; we only care about groups 1 onward
                ret.add(new Text(m.group(i)));
            }
        } else {
            throw new IllegalStateException("Invalid line: line does not match regex (line #" + currLine + ", regex=\""
                            + regex + "\"; line=\"" + line + "\"");
        }

        return ret;
    }

    @Override
    public void reset() {
        super.reset();
        numLinesSkipped = 0;
    }

    @Override
    public Record nextRecord() {
        List<Writable> next = next();
        URI uri = (locations == null || locations.length < 1 ? null : locations[splitIndex]);
        RecordMetaData meta = new RecordMetaDataLine(this.lineIndex - 1, uri, RegexLineRecordReader.class); //-1 as line number has been incremented already...
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
}
