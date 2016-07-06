/*
 *
 *  *
 *  *  * Copyright 2015 Skymind,Inc.
 *  *  *
 *  *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *  *    you may not use this file except in compliance with the License.
 *  *  *    You may obtain a copy of the License at
 *  *  *
 *  *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *  *
 *  *  *    Unless required by applicable law or agreed to in writing, software
 *  *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *  *    See the License for the specific language governing permissions and
 *  *  *    limitations under the License.
 *  *
 *
 */

package org.canova.api.records.reader.impl;


import org.canova.api.conf.Configuration;
import org.canova.api.records.reader.BaseRecordReader;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.split.InputSplit;
import org.canova.api.writable.Writable;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * Collection record reader for sequences.
 * Mainly used for testing.
 *
 * @author Alex Black
 */
public class CollectionSequenceRecordReader extends BaseRecordReader implements SequenceRecordReader {
    private Iterator<? extends Collection<? extends Collection<Writable>>> records;
    private final Collection<? extends Collection<? extends Collection<Writable>>> original;

    /**
     *
     * @param records    Collection of sequences. For example, List<List<List<Writable>>> where the inner  two lists
     *                   are a sequence, and the outer list/collection is a list of sequences
     */
    public CollectionSequenceRecordReader(Collection<? extends Collection<? extends Collection<Writable>>> records) {
        this.records = records.iterator();
        this.original = records;
    }

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {

    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        initialize(split);
    }

    @Override
    public Collection<Writable> next() {
        throw new UnsupportedOperationException("next() not supported for CollectionSequencRecordReader; use sequenceRecord()");
    }

    @Override
    public boolean hasNext() {
        return records.hasNext();
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

    @Override
    public List<String> getLabels(){
        return null;
    }

    @Override
    public void reset() {
        this.records = original.iterator();
    }

    @Override
    public Collection<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException("Generating records from DataInputStream not supported for SequenceCollectionRecordReader");
    }


    @Override
    @SuppressWarnings("unchecked")
    public Collection<Collection<Writable>> sequenceRecord() {
        Collection<Collection<Writable>> record = (Collection<Collection<Writable>>)records.next();
        invokeListeners(record);
        return record;
    }

    @Override
    public Collection<Collection<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException("Generating records from DataInputStream not supported for SequenceCollectionRecordReader");
    }
}
