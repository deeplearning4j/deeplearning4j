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

package org.datavec.api.records.reader.impl.collection;


import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataIndex;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.*;

/**
 * Collection record reader for sequences.
 * Mainly used for testing.
 *
 * @author Alex Black
 */
public class CollectionSequenceRecordReader extends BaseRecordReader implements SequenceRecordReader {
    private Iterator<? extends Collection<? extends Collection<Writable>>> records;
    private final Collection<? extends Collection<? extends Collection<Writable>>> original;
    private int count = 0;

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
    public List<Writable> next() {
        throw new UnsupportedOperationException(
                        "next() not supported for CollectionSequencRecordReader; use sequenceRecord()");
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
    public List<String> getLabels() {
        return null;
    }

    @Override
    public void reset() {
        this.records = original.iterator();
        this.count = 0;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException(
                        "Generating records from DataInputStream not supported for SequenceCollectionRecordReader");
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


    @Override
    @SuppressWarnings("unchecked")
    public List<List<Writable>> sequenceRecord() {
        List<List<Writable>> record = toList(records.next());
        invokeListeners(record);
        count++;
        return record;
    }

    @Override
    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException(
                        "Generating records from DataInputStream not supported for SequenceCollectionRecordReader");
    }

    @Override
    public SequenceRecord nextSequence() {
        return new org.datavec.api.records.impl.SequenceRecord(sequenceRecord(),
                        new RecordMetaDataIndex(count - 1, null, CollectionSequenceRecordReader.class));
    }

    @Override
    public SequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return loadSequenceFromMetaData(Collections.singletonList(recordMetaData)).get(0);
    }

    @Override
    public List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        Set<Integer> toLoad = new LinkedHashSet<>();
        for (RecordMetaData recordMetaData : recordMetaDatas) {
            if (!(recordMetaData instanceof RecordMetaDataIndex)) {
                throw new IllegalArgumentException("Expected RecordMetaDataIndex; got: " + recordMetaData);
            }
            long idx = ((RecordMetaDataIndex) recordMetaData).getIndex();
            if (idx >= original.size()) {
                throw new IllegalStateException(
                                "Cannot get index " + idx + " from collection: contains " + original + " elements");
            }
            toLoad.add((int) idx);
        }

        List<SequenceRecord> out = new ArrayList<>();
        Iterator<? extends Collection<? extends Collection<Writable>>> iter = original.iterator();
        int i = 0;
        while (iter.hasNext()) {
            Collection<? extends Collection<Writable>> c = iter.next();
            if (!toLoad.contains(i++)) {
                continue;
            }
            List<List<Writable>> record = toList(c);
            SequenceRecord r = new org.datavec.api.records.impl.SequenceRecord(record,
                            new RecordMetaDataIndex(i - 1, null, CollectionSequenceRecordReader.class));
            out.add(r);
        }
        return out;
    }

    private static List<List<Writable>> toList(Collection<? extends Collection<Writable>> next) {
        List<List<Writable>> record = new ArrayList<>();
        for (Collection<Writable> c : next) {
            record.add(new ArrayList<>(c));
        }
        return record;
    }
}
