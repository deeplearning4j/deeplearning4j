/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.api.records.reader.impl.filebatch;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;
import org.nd4j.common.loader.FileBatch;
import org.nd4j.common.base.Preconditions;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.Collection;
import java.util.List;

/**
 * FileBatchSequenceRecordReader reads the files contained in a {@link FileBatch} using the specified SequenceRecordReader.<br>
 * Specifically, the {@link SequenceRecordReader#sequenceRecord(URI, DataInputStream)} } method of the underlying sequence
 * reader is used to load files.<br>
 * For example, if the FileBatch was constructed using csv sequence files (each file represents one example),
 * FileBatchSequencRecordReader could be used with CSVSequenceRecordReader. For example:<br>
 * <pre>
 * {@code
 * List<File> fileList = ...;
 * FileBatch fb = FileBatch.forFiles(imgFiles);
 * SequenceRecordReader rr = new CSVSequenceRecordReader();
 * FileBatchSequenceRecordReader fbrr = new FileBatchSequenceRecordReader(rr, fb);
 * }
 * </pre>
 *
 * @author Alex Black
 */
public class FileBatchSequenceRecordReader implements SequenceRecordReader {

    private final SequenceRecordReader recordReader;
    private final FileBatch fileBatch;
    private int position = 0;

    /**
     * @param seqRR     Underlying record reader to read files from
     * @param fileBatch File batch to read files from
     */
    public FileBatchSequenceRecordReader(SequenceRecordReader seqRR, FileBatch fileBatch){
        this.recordReader = seqRR;
        this.fileBatch = fileBatch;
    }

    @Override
    public List<List<Writable>> sequenceRecord() {
        Preconditions.checkState(hasNext(), "No next element");

        byte[] fileBytes = fileBatch.getFileBytes().get(position);
        String origPath = fileBatch.getOriginalUris().get(position);

        List<List<Writable>> out;
        try {
            out = recordReader.sequenceRecord(URI.create(origPath), new DataInputStream(new ByteArrayInputStream(fileBytes)));
        } catch (IOException e){
            throw new RuntimeException("Error reading from file bytes");
        }

        position++;
        return out;
    }

    @Override
    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        return recordReader.sequenceRecord(uri, dataInputStream);
    }

    @Override
    public SequenceRecord nextSequence() {
        return new org.datavec.api.records.impl.SequenceRecord(sequenceRecord(), null);
    }

    @Override
    public SequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return recordReader.loadSequenceFromMetaData(recordMetaData);
    }

    @Override
    public List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        return recordReader.loadSequenceFromMetaData(recordMetaDatas);
    }

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        //No op
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        //No op
    }

    @Override
    public boolean batchesSupported() {
        return false;
    }

    @Override
    public List<List<Writable>> next(int num) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public List<Writable> next() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public boolean hasNext() {
        return position < fileBatch.getFileBytes().size();
    }

    @Override
    public List<String> getLabels() {
        return recordReader.getLabels();
    }

    @Override
    public void reset() {
        position = 0;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public Record nextRecord() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public List<RecordListener> getListeners() {
        return recordReader.getListeners();
    }

    @Override
    public void setListeners(RecordListener... listeners) {
        recordReader.setListeners(listeners);
    }

    @Override
    public void setListeners(Collection<RecordListener> listeners) {
        recordReader.setListeners(listeners);
    }

    @Override
    public void close() throws IOException {
        recordReader.close();
    }

    @Override
    public void setConf(Configuration conf) {
        recordReader.setConf(conf);
    }

    @Override
    public Configuration getConf() {
        return recordReader.getConf();
    }
}
