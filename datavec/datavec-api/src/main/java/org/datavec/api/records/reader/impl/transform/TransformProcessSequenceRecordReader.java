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

package org.datavec.api.records.reader.impl.transform;

import lombok.AllArgsConstructor;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.writable.Writable;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.Collection;
import java.util.List;

/**
 * This wraps a {@link SequenceRecordReader} with a {@link TransformProcess}
 * which will allow every {@link Record} returned from the {@link SequenceRecordReader}
 * to be transformed before being returned.
 *
 * @author Adam Gibson
 */
@AllArgsConstructor
public class TransformProcessSequenceRecordReader implements SequenceRecordReader {

    protected SequenceRecordReader sequenceRecordReader;
    protected TransformProcess transformProcess;



    /**
     * Set the configuration to be used by this object.
     *
     * @param conf
     */
    @Override
    public void setConf(Configuration conf) {
        sequenceRecordReader.setConf(conf);
    }

    /**
     * Return the configuration used by this object.
     */
    @Override
    public Configuration getConf() {
        return sequenceRecordReader.getConf();
    }

    /**
     * Returns a sequence record.
     *
     * @return a sequence of records
     */
    @Override
    public List<List<Writable>> sequenceRecord() {
        return transformProcess.executeSequence(sequenceRecordReader.sequenceRecord());
    }

    @Override
    public boolean batchesSupported() {
        return false;
    }

    @Override
    public List<List<Writable>> next(int num) {
        throw new UnsupportedOperationException();
    }

    /**
     * Load a sequence record from the given DataInputStream
     * Unlike {@link #next()} the internal state of the RecordReader is not modified
     * Implementations of this method should not close the DataInputStream
     *
     * @param uri
     * @param dataInputStream
     * @throws IOException if error occurs during reading from the input stream
     */
    @Override
    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        return transformProcess.executeSequence(sequenceRecordReader.sequenceRecord(uri, dataInputStream));
    }

    /**
     * Similar to {@link #sequenceRecord()}, but returns a {@link Record} object, that may include metadata such as the source
     * of the data
     *
     * @return next sequence record
     */
    @Override
    public SequenceRecord nextSequence() {
        SequenceRecord next = sequenceRecordReader.nextSequence();
        next.setSequenceRecord(transformProcess.executeSequence(next.getSequenceRecord()));
        return next;
    }

    /**
     * Load a single sequence record from the given {@link RecordMetaData} instance<br>
     * Note: that for data that isn't splittable (i.e., text data that needs to be scanned/split), it is more efficient to
     * load multiple records at once using {@link #loadSequenceFromMetaData(List)}
     *
     * @param recordMetaData Metadata for the sequence record that we want to load from
     * @return Single sequence record for the given RecordMetaData instance
     * @throws IOException If I/O error occurs during loading
     */
    @Override
    public SequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException {
        SequenceRecord next = sequenceRecordReader.loadSequenceFromMetaData(recordMetaData);
        next.setSequenceRecord(transformProcess.executeSequence(next.getSequenceRecord()));
        return next;
    }

    /**
     * Load multiple sequence records from the given a list of {@link RecordMetaData} instances<br>
     *
     * @param recordMetaDatas Metadata for the records that we want to load from
     * @return Multiple sequence record for the given RecordMetaData instances
     * @throws IOException If I/O error occurs during loading
     */
    @Override
    public List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        return null;
    }

    /**
     * Called once at initialization.
     *
     * @param split the split that defines the range of records to read
     * @throws IOException
     * @throws InterruptedException
     */
    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {

    }

    /**
     * Called once at initialization.
     *
     * @param conf  a configuration for initialization
     * @param split the split that defines the range of records to read
     * @throws IOException
     * @throws InterruptedException
     */
    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {

    }

    /**
     * Get the next record
     *
     * @return
     */
    @Override
    public List<Writable> next() {
        return transformProcess.execute(sequenceRecordReader.next());
    }

    /**
     * Whether there are anymore records
     *
     * @return
     */
    @Override
    public boolean hasNext() {
        return sequenceRecordReader.hasNext();
    }

    /**
     * List of label strings
     *
     * @return
     */
    @Override
    public List<String> getLabels() {
        return sequenceRecordReader.getLabels();
    }

    /**
     * Reset record reader iterator
     *
     * @return
     */
    @Override
    public void reset() {
        sequenceRecordReader.reset();
    }

    @Override
    public boolean resetSupported() {
        return sequenceRecordReader.resetSupported();
    }

    /**
     * Load the record from the given DataInputStream
     * Unlike {@link #next()} the internal state of the RecordReader is not modified
     * Implementations of this method should not close the DataInputStream
     *
     * @param uri
     * @param dataInputStream
     * @throws IOException if error occurs during reading from the input stream
     */
    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        return transformProcess.execute(sequenceRecordReader.record(uri, dataInputStream));
    }

    /**
     * Similar to {@link #next()}, but returns a {@link Record} object, that may include metadata such as the source
     * of the data
     *
     * @return next record
     */
    @Override
    public Record nextRecord() {
        Record next = sequenceRecordReader.nextRecord();
        next.setRecord(transformProcess.execute(next.getRecord()));
        return next;
    }

    /**
     * Load a single record from the given {@link RecordMetaData} instance<br>
     * Note: that for data that isn't splittable (i.e., text data that needs to be scanned/split), it is more efficient to
     * load multiple records at once using {@link #loadFromMetaData(List)}
     *
     * @param recordMetaData Metadata for the record that we want to load from
     * @return Single record for the given RecordMetaData instance
     * @throws IOException If I/O error occurs during loading
     */
    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        Record load = sequenceRecordReader.loadFromMetaData(recordMetaData);
        load.setRecord(transformProcess.execute(load.getRecord()));
        return load;
    }

    /**
     * Load multiple records from the given a list of {@link RecordMetaData} instances<br>
     *
     * @param recordMetaDatas Metadata for the records that we want to load from
     * @return Multiple records for the given RecordMetaData instances
     * @throws IOException If I/O error occurs during loading
     */
    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        List<Record> records = sequenceRecordReader.loadFromMetaData(recordMetaDatas);
        for (Record record : records)
            record.setRecord(transformProcess.execute(record.getRecord()));
        return records;
    }

    /**
     * Get the record listeners for this record reader.
     */
    @Override
    public List<RecordListener> getListeners() {
        return sequenceRecordReader.getListeners();
    }

    /**
     * Set the record listeners for this record reader.
     *
     * @param listeners
     */
    @Override
    public void setListeners(RecordListener... listeners) {
        sequenceRecordReader.setListeners(listeners);
    }

    /**
     * Set the record listeners for this record reader.
     *
     * @param listeners
     */
    @Override
    public void setListeners(Collection<RecordListener> listeners) {
        sequenceRecordReader.setListeners(listeners);
    }

    /**
     * Closes this stream and releases any system resources associated
     * with it. If the stream is already closed then invoking this
     * method has no effect.
     * <p>
     * <p> As noted in {@link AutoCloseable#close()}, cases where the
     * close may fail require careful attention. It is strongly advised
     * to relinquish the underlying resources and to internally
     * <em>mark</em> the {@code Closeable} as closed, prior to throwing
     * the {@code IOException}.
     *
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void close() throws IOException {
        sequenceRecordReader.close();
    }
}
