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
import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;
import org.nd4j.common.loader.FileBatch;
import org.nd4j.common.base.Preconditions;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * FileBatchRecordReader reads the files contained in a {@link FileBatch} using the specified RecordReader.<br>
 * Specifically, the {@link RecordReader#record(URI, DataInputStream)} method of the underlying reader is used to
 * load files.<br>
 * For example, if the FileBatch was constructed using image files (png, jpg etc), FileBatchRecordReader could be used
 * with ImageRecordReader. For example:<br>
 * <pre>
 * {@code
 * List<File> imgFiles = ...;
 * FileBatch fb = FileBatch.forFiles(imgFiles);
 * PathLabelGenerator labelMaker = new ParentPathLabelGenerator();
 * ImageRecordReader rr = new ImageRecordReader(32, 32, 1, labelMaker);
 * rr.setLabels(Arrays.asList("class0", "class1"));
 * FileBatchRecordReader fbrr = new FileBatchRecordReader(rr, fb);
 * }
 * </pre>
 *
 * @author Alex Black
 */
public class FileBatchRecordReader implements RecordReader {

    private final RecordReader recordReader;
    private final FileBatch fileBatch;
    private int position = 0;

    /**
     * @param rr        Underlying record reader to read files from
     * @param fileBatch File batch to read files from
     */
    public FileBatchRecordReader(RecordReader rr, FileBatch fileBatch){
        this.recordReader = rr;
        this.fileBatch = fileBatch;
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
        List<List<Writable>> out = new ArrayList<>(Math.min(num, 10000));
        for( int i=0; i<num && hasNext(); i++ ){
            out.add(next());
        }
        return out;
    }

    @Override
    public List<Writable> next() {
        Preconditions.checkState(hasNext(), "No next element");

        byte[] fileBytes = fileBatch.getFileBytes().get(position);
        String origPath = fileBatch.getOriginalUris().get(position);

        List<Writable> out;
        try {
            out = recordReader.record(URI.create(origPath), new DataInputStream(new ByteArrayInputStream(fileBytes)));
        } catch (IOException e){
            throw new RuntimeException("Error reading from file bytes");
        }

        position++;
        return out;
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
        return new org.datavec.api.records.impl.Record(next(), null);
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return recordReader.loadFromMetaData(recordMetaData);
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        return recordReader.loadFromMetaData(recordMetaDatas);
    }

    @Override
    public List<RecordListener> getListeners() {
        return null;
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
