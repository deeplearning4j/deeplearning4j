/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.api.records.reader.impl.csv;

import org.datavec.api.records.Record;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.writable.Writable;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class CSVLineSequenceRecordReader extends CSVRecordReader implements SequenceRecordReader {

    /**
     * Default settings: skip 0 lines, use ',' as the delimiter, and '"' for quotes
     */
    public CSVLineSequenceRecordReader(){
        this(0, DEFAULT_DELIMITER, DEFAULT_QUOTE);
    }

    /**
     * Skip lines and use delimiter
     * @param skipNumLines the number of lines to skip
     * @param delimiter the delimiter
     */
    public CSVLineSequenceRecordReader(int skipNumLines, char delimiter) {
        this(skipNumLines, delimiter, '\"');
    }

    /**
     * Skip lines, use delimiter, and strip quotes
     * @param skipNumLines the number of lines to skip
     * @param delimiter the delimiter
     * @param quote the quote to strip
     */
    public CSVLineSequenceRecordReader(int skipNumLines, char delimiter, char quote) {
        super(skipNumLines, delimiter, quote);
    }

    @Override
    public List<List<Writable>> sequenceRecord() {
        return nextSequence().getSequenceRecord();
    }

    @Override
    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        List<Writable> l = record(uri, dataInputStream);
        List<List<Writable>> out = new ArrayList<>();
        for(Writable w : l){
            out.add(Collections.singletonList(w));
        }
        return out;
    }

    @Override
    public SequenceRecord nextSequence() {
        return convert(super.nextRecord());
    }

    @Override
    public SequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return convert(super.loadFromMetaData(recordMetaData));
    }

    @Override
    public List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        List<Record> toConvert = super.loadFromMetaData(recordMetaDatas);
        List<SequenceRecord> out = new ArrayList<>();
        for(Record r  : toConvert){
            out.add(convert(r));
        }
        return out;
    }

    protected SequenceRecord convert(Record r){
        List<Writable> line = r.getRecord();
        List<List<Writable>> out = new ArrayList<>();
        for(Writable w : line){
            out.add(Collections.singletonList(w));
        }
        return new org.datavec.api.records.impl.SequenceRecord(out, r.getMetaData());
    }
}
