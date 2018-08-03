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

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.FileRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.URI;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * RegexSequenceRecordReader: Read an entire file (as a sequence), one line at a time and
 * split each line into fields using a regex.
 * Specifically, we are using {@link Pattern} and {@link Matcher} to do the splitting into groups
 *
 * Example: Data in format "2016-01-01 23:59:59.001 1 DEBUG First entry message!"<br>
 * using regex String "(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\.\\d{3}) (\\d+) ([A-Z]+) (.*)"<br>
 * would be split into 4 Text writables: ["2016-01-01 23:59:59.001", "1", "DEBUG", "First entry message!"]<br>
 *
 * Note: RegexSequenceRecordReader supports multiple error handling modes, via {@link LineErrorHandling}. Invalid
 * lines that don't match the provided regex can result in an exception (FailOnInvalid), can be skipped silently (SkipInvalid),
 * or skip invalid but log a warning (SkipInvalidWithWarning)
 *
 * @author Alex Black
 */
public class RegexSequenceRecordReader extends FileRecordReader implements SequenceRecordReader {
    public static final String SKIP_NUM_LINES = NAME_SPACE + ".skipnumlines";
    public static final Charset DEFAULT_CHARSET = Charset.forName("UTF-8");
    public static final LineErrorHandling DEFAULT_ERROR_HANDLING = LineErrorHandling.FailOnInvalid;

    /**Error handling mode: How should invalid lines (i.e., those that don't match the provided regex) be handled?<br>
     * FailOnInvalid: Throw an IllegalStateException when an invalid line is found<br>
     * SkipInvalid: Skip invalid lines (quietly, with no warning)<br>
     * SkipInvalidWithWarning: Skip invalid lines, but log a warning<br>
     */
    public enum LineErrorHandling {
        FailOnInvalid, SkipInvalid, SkipInvalidWithWarning
    };

    public static final Logger LOG = LoggerFactory.getLogger(RegexSequenceRecordReader.class);

    private String regex;
    private int skipNumLines;
    private Pattern pattern;
    private transient Charset charset;
    private LineErrorHandling errorHandling;

    public RegexSequenceRecordReader(String regex, int skipNumLines) {
        this(regex, skipNumLines, DEFAULT_CHARSET, DEFAULT_ERROR_HANDLING);
    }

    public RegexSequenceRecordReader(String regex, int skipNumLines, Charset encoding,
                    LineErrorHandling errorHandling) {
        this.regex = regex;
        this.skipNumLines = skipNumLines;
        this.pattern = Pattern.compile(regex);
        this.charset = encoding;
        this.errorHandling = errorHandling;
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        super.initialize(conf, split);
        this.skipNumLines = conf.getInt(SKIP_NUM_LINES, this.skipNumLines);
    }

    public List<List<Writable>> sequenceRecord() {
        return nextSequence().getSequenceRecord();
    }

    @Override
    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        String fileContents = IOUtils.toString(new BufferedInputStream(dataInputStream), charset.name());
        return loadSequence(fileContents, uri);
    }

    private List<List<Writable>> loadSequence(String fileContents, URI uri) {
        String[] lines = fileContents.split("(\r\n)|\n"); //TODO this won't work if regex allows for a newline

        int numLinesSkipped = 0;
        List<List<Writable>> out = new ArrayList<>();
        int lineCount = 0;
        for (String line : lines) {
            lineCount++;
            if (numLinesSkipped < skipNumLines) {
                numLinesSkipped++;
                continue;
            }
            //Split line using regex matcher
            Matcher m = pattern.matcher(line);
            List<Writable> timeStep;
            if (m.matches()) {
                int count = m.groupCount();
                timeStep = new ArrayList<>(count);
                for (int i = 1; i <= count; i++) { //Note: Matcher.group(0) is the entire sequence; we only care about groups 1 onward
                    timeStep.add(new Text(m.group(i)));
                }
            } else {
                switch (errorHandling) {
                    case FailOnInvalid:
                        throw new IllegalStateException(
                                        "Invalid line: line does not match regex (line #" + lineCount + ", uri=\"" + uri
                                                        + "\"), " + "\", regex=" + regex + "\"; line=\"" + line + "\"");
                    case SkipInvalid:
                        continue;
                    case SkipInvalidWithWarning:
                        String warnMsg = "Skipping invalid line: line does not match regex (line #" + lineCount
                                        + ", uri=\"" + uri + "\"), " + "\"; line=\"" + line + "\"";
                        LOG.warn(warnMsg);
                        continue;
                    default:
                        throw new RuntimeException("Unknown error handling mode: " + errorHandling);
                }
            }
            out.add(timeStep);
        }

        return out;
    }

    @Override
    public void reset() {
        super.reset();
    }

    @Override
    public SequenceRecord nextSequence() {
        File next = this.nextFile();

        String fileContents;
        try {
            fileContents = FileUtils.readFileToString(next, charset.name());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        List<List<Writable>> sequence = loadSequence(fileContents, next.toURI());
        return new org.datavec.api.records.impl.SequenceRecord(sequence,
                        new RecordMetaDataURI(next.toURI(), RegexSequenceRecordReader.class));
    }

    @Override
    public SequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return loadSequenceFromMetaData(Collections.singletonList(recordMetaData)).get(0);
    }

    @Override
    public List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        List<SequenceRecord> out = new ArrayList<>();
        for (RecordMetaData meta : recordMetaDatas) {
            File next = new File(meta.getURI());
            URI uri = next.toURI();
            String fileContents = FileUtils.readFileToString(next, charset.name());
            List<List<Writable>> sequence = loadSequence(fileContents, uri);
            out.add(new org.datavec.api.records.impl.SequenceRecord(sequence, meta));
        }
        return out;
    }

    private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException {
        ois.defaultReadObject();
        String s = ois.readUTF();
        charset = Charset.forName(s);
    }

    private void writeObject(ObjectOutputStream oos) throws IOException {
        oos.defaultWriteObject();
        oos.writeUTF(charset.name());
    }
}
