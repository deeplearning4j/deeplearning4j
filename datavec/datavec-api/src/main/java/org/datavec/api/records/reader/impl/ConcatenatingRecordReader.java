/*
 *  * Copyright 2017 Skymind, Inc.
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

package org.datavec.api.records.reader.impl;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;

/**
 * Combine multiple readers into a single reader. Records are read sequentially - thus if the first reader has
 * 100 records, and the second reader has 200 records, ConcatenatingRecordReader will have 300 records.
 *
 * See also {@link ComposableRecordReader} for a version that combines each record from underlying readers.
 *
 * @author Alex Black
 */
public class ConcatenatingRecordReader extends BaseRecordReader {

    private RecordReader[] readers;

    public ConcatenatingRecordReader(RecordReader... readers) {
        this.readers = readers;
    }

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {

    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {

    }

    @Override
    public List<Writable> next() {
        List<Writable> out = null;
        for( RecordReader rr : readers){
            if(rr.hasNext()){
                out = rr.next();
                break;
            }
        }
        invokeListeners(out);
        return out;
    }

    @Override
    public boolean hasNext() {
        for (RecordReader reader : readers) {
            if(reader.hasNext()){
                return true;
            }
        }
        return false;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public void close() throws IOException {
        for (RecordReader reader : readers)
            reader.close();
    }

    @Override
    public void setConf(Configuration conf) {
        for (RecordReader reader : readers) {
            reader.setConf(conf);
        }
    }

    @Override
    public Configuration getConf() {
        return readers[0].getConf();
    }

    @Override
    public void reset() {
        for (RecordReader reader : readers)
            reader.reset();
    }

    @Override
    public boolean resetSupported() {
        for(RecordReader rr : readers){
            if(!rr.resetSupported()){
                return false;
            }
        }
        return true;
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException(
                        "Generating records from DataInputStream not supported for ComposableRecordReader");
    }

    @Override
    public Record nextRecord() {
        return new org.datavec.api.records.impl.Record(next(), null);
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        throw new UnsupportedOperationException("Loading from metadata not yet implemented");
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        throw new UnsupportedOperationException("Loading from metadata not yet implemented");
    }


}
