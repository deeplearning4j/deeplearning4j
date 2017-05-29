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

package org.datavec.hadoop.records.reader;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataIndex;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;
import org.datavec.hadoop.records.reader.mapfile.MapFileReader;
import org.datavec.hadoop.records.reader.mapfile.index.LongIndexToKey;
import org.datavec.hadoop.records.reader.mapfile.record.SequenceRecordWritable;
import org.datavec.hadoop.records.reader.mapfile.record.SequenceRecordWritableCreator;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.Collection;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Created by Alex on 29/05/2017.
 */
public class MapFileSequenceRecordReader implements SequenceRecordReader {

    private final MapFileReader<SequenceRecordWritable> mapFileReader;
    private final long numSequences;
    private long position;

    public MapFileSequenceRecordReader(String path) throws Exception {
        this(new MapFileReader(path, new LongIndexToKey(), new SequenceRecordWritableCreator()));
    }

    public MapFileSequenceRecordReader(MapFileReader mapFileReader){
        this.mapFileReader = mapFileReader;
        this.numSequences = mapFileReader.numRecords();
    }

    @Override
    public void setConf(Configuration conf) {

    }

    @Override
    public Configuration getConf() {
        return null;
    }

    @Override
    public List<List<Writable>> sequenceRecord() {
        return nextSequence().getSequenceRecord();
    }

    @Override
    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException("MapFileSequenceRecordReader: does not support reading from streams");
    }

    @Override
    public SequenceRecord nextSequence() {
        if(!hasNext()){
            throw new NoSuchElementException();
        }

        SequenceRecordWritable seq;
        long currIdx = position++;
        try{
            seq = mapFileReader.getRecord(currIdx);
        } catch (IOException e){
            throw new RuntimeException(e);
        }

        return new org.datavec.api.records.impl.SequenceRecord(seq.getSequenceRecord(), new RecordMetaDataIndex(currIdx, null, MapFileSequenceRecordReader.class));  //TODO metadata
    }

    @Override
    public SequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return null;
    }

    @Override
    public List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        throw new UnsupportedOperationException("Not yet implemented");
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
    public List<Writable> next(int num) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<Writable> next() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean hasNext() {
        return position < numSequences;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public void reset() {
        position = 0;
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public Record nextRecord() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        throw new UnsupportedOperationException();
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
        if(mapFileReader != null){
            mapFileReader.close();
        }
    }
}
