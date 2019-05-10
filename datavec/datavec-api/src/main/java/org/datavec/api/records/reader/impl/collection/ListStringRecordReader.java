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

package org.datavec.api.records.reader.impl.collection;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.ListStringSplit;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Iterates through a list of strings return a record.
 * Only accepts an @link {ListStringInputSplit} as input.
 *
 * @author Adam Gibson
 */
public class ListStringRecordReader extends BaseRecordReader {
    private List<List<String>> delimitedData;
    private Iterator<List<String>> dataIter;
    private Configuration conf;

    /**
     * Called once at initialization.
     *
     * @param split the split that defines the range of records to read
     * @throws IOException
     * @throws InterruptedException
     */
    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        if (split instanceof ListStringSplit) {
            ListStringSplit listStringSplit = (ListStringSplit) split;
            delimitedData = listStringSplit.getData();
            dataIter = delimitedData.iterator();
        } else {
            throw new IllegalArgumentException("Illegal type of input split " + split.getClass().getName());
        }
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
        initialize(split);
    }

    /**
     * Get the next record
     *
     * @return The list of next record
     */
    @Override
    public List<Writable> next() {
        List<String> next = dataIter.next();
        invokeListeners(next);
        List<Writable> ret = new ArrayList<>();
        for (String s : next)
            ret.add(new Text(s));
        return ret;
    }

    /**
     * Check whether there are anymore records
     *
     * @return Whether there are more records
     */
    @Override
    public boolean hasNext() {
        return dataIter.hasNext();
    }

    /**
     * List of label strings
     *
     * @return
     */
    @Override
    public List<String> getLabels() {
        return null;
    }

    /**
     * Reset record reader iterator
     *
     */
    @Override
    public void reset() {
        dataIter = delimitedData.iterator();
    }

    @Override
    public boolean resetSupported() {
        return true;
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
        return null;
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

    }

    /**
     * Set the configuration to be used by this object.
     *
     * @param conf
     */
    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    /**
     * Return the configuration used by this object.
     */
    @Override
    public Configuration getConf() {
        return conf;
    }
}
