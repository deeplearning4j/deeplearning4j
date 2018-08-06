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

package org.datavec.api.records.reader.impl.csv;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;
import org.nd4j.base.Preconditions;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.Collection;
import java.util.List;
import java.util.NoSuchElementException;

public class CSVMultiSequenceRecordReader extends CSVRecordReader implements SequenceRecordReader {

    public enum Mode {
        CONCAT,
        EQUAL_LENGTH,
        PAD
    }

    private String sequenceSeparatorRegex;
    private Mode mode;

    public CSVMultiSequenceRecordReader(String sequenceSeparatorRegex, char elementDelimiter, Mode mode){
        this(0, DEFAULT_DELIMITER, DEFAULT_QUOTE, sequenceSeparatorRegex, elementDelimiter, mode, null);
    }

    public CSVMultiSequenceRecordReader(int skipNumLines, char delimiter, char quote, String sequenceSeparatorRegex, char elementDelimiter, Mode mode, Writable padValue){
        super(skipNumLines, delimiter, quote);
        Preconditions.checkState(mode != Mode.PAD || padValue != null, "Cannot use Mode.PAD with a null padding value. " +
                "Padding value must be passed to constructor ");
        this.sequenceSeparatorRegex = sequenceSeparatorRegex;
        this.mode = mode;
    }


    @Override
    public List<List<Writable>> sequenceRecord() {
        return nextSequence().getSequenceRecord();
    }

    @Override
    public SequenceRecord nextSequence() {
        if(!hasNext())
            throw new NoSuchElementException("No next element");

        while(super.hasNext()){
            String line = 
        }
    }

    @Override
    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        return null;
    }

    @Override
    public SequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return null;
    }

    @Override
    public List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        return null;
    }

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {

    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {

    }

    @Override
    public boolean batchesSupported() {
        return false;
    }

    @Override
    public List<List<Writable>> next(int num) {
        return null;
    }

    @Override
    public List<Writable> next() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return false;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public void reset() {

    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        return null;
    }

    @Override
    public Record nextRecord() {
        return null;
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return null;
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        return null;
    }

    @Override
    public List<RecordListener> getListeners() {
        return null;
    }

    @Override
    public void setListeners(RecordListener... listeners) {

    }

    @Override
    public void setListeners(Collection<RecordListener> listeners) {

    }

    @Override
    public void close() throws IOException {

    }

    @Override
    public void setConf(Configuration conf) {

    }

    @Override
    public Configuration getConf() {
        return null;
    }
}
