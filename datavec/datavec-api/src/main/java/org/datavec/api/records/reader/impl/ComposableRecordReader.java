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
 * @author sonali
 */
/**
RecordReader for each pipeline. Individual record is a concatenation of the two collections.
        Create a recordreader that takes recordreaders and iterates over them and concatenates them
        hasNext would be the & of all the recordreaders
        concatenation would be next & addAll on the collection
        return one record
 */
public class ComposableRecordReader extends BaseRecordReader {

    private RecordReader[] readers;

    public ComposableRecordReader(RecordReader... readers) {
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
        List<Writable> ret = new ArrayList<>();
        if (this.hasNext()) {
            for (RecordReader reader : readers) {
                ret.addAll(reader.next());
            }
        }
        invokeListeners(ret);
        return ret;
    }

    @Override
    public boolean hasNext() {
        boolean readersHasNext = true;
        for (RecordReader reader : readers) {
            readersHasNext = readersHasNext && reader.hasNext();
        }
        return readersHasNext;
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
        for (RecordReader reader : readers) {
            return reader.getConf();
        }
        return null;
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
