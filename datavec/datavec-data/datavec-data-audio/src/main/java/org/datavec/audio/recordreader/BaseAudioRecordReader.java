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

package org.datavec.audio.recordreader;

import org.apache.commons.io.FileUtils;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.split.BaseInputSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.InputStreamInputSplit;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;

import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * Base audio file loader
 * @author Adam Gibson
 */
public abstract class BaseAudioRecordReader extends BaseRecordReader {
    private Iterator<File> iter;
    private List<Writable> record;
    private boolean hitImage = false;
    private boolean appendLabel = false;
    private List<String> labels = new ArrayList<>();
    private Configuration conf;
    protected InputSplit inputSplit;

    public BaseAudioRecordReader() {}

    public BaseAudioRecordReader(boolean appendLabel, List<String> labels) {
        this.appendLabel = appendLabel;
        this.labels = labels;
    }

    public BaseAudioRecordReader(List<String> labels) {
        this.labels = labels;
    }

    public BaseAudioRecordReader(boolean appendLabel) {
        this.appendLabel = appendLabel;
    }

    protected abstract List<Writable> loadData(File file, InputStream inputStream) throws IOException;

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        inputSplit = split;
        if (split instanceof BaseInputSplit) {
            URI[] locations = split.locations();
            if (locations != null && locations.length >= 1) {
                if (locations.length > 1) {
                    List<File> allFiles = new ArrayList<>();
                    for (URI location : locations) {
                        File iter = new File(location);
                        if (iter.isDirectory()) {
                            Iterator<File> allFiles2 = FileUtils.iterateFiles(iter, null, true);
                            while (allFiles2.hasNext())
                                allFiles.add(allFiles2.next());
                        }

                        else
                            allFiles.add(iter);
                    }

                    iter = allFiles.iterator();
                } else {
                    File curr = new File(locations[0]);
                    if (curr.isDirectory())
                        iter = FileUtils.iterateFiles(curr, null, true);
                    else
                        iter = Collections.singletonList(curr).iterator();
                }
            }
        }


        else if (split instanceof InputStreamInputSplit) {
            record = new ArrayList<>();
            InputStreamInputSplit split2 = (InputStreamInputSplit) split;
            InputStream is = split2.getIs();
            URI[] locations = split2.locations();
            if (appendLabel) {
                Path path = Paths.get(locations[0]);
                String parent = path.getParent().toString();
                record.add(new DoubleWritable(labels.indexOf(parent)));
            }

            is.close();
        }

    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        this.conf = conf;
        this.appendLabel = conf.getBoolean(APPEND_LABEL, false);
        this.labels = new ArrayList<>(conf.getStringCollection(LABELS));
        initialize(split);
    }

    @Override
    public List<Writable> next() {
        if (iter != null) {
            File next = iter.next();
            invokeListeners(next);
            try {
                return loadData(next, null);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } else if (record != null) {
            hitImage = true;
            return record;
        }

        throw new IllegalStateException("Indeterminant state: record must not be null, or a file iterator must exist");
    }

    @Override
    public boolean hasNext() {
        if (iter != null) {
            return iter.hasNext();
        } else if (record != null) {
            return !hitImage;
        }
        throw new IllegalStateException("Indeterminant state: record must not be null, or a file iterator must exist");
    }


    @Override
    public void close() throws IOException {

    }

    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    @Override
    public Configuration getConf() {
        return conf;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }


    @Override
    public void reset() {
        if (inputSplit == null)
            throw new UnsupportedOperationException("Cannot reset without first initializing");
        try {
            initialize(inputSplit);
        } catch (Exception e) {
            throw new RuntimeException("Error during LineRecordReader reset", e);
        }
    }

    @Override
    public boolean resetSupported(){
        if(inputSplit == null){
            return false;
        }
        return inputSplit.resetSupported();
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        invokeListeners(uri);
        try {
            return loadData(null, dataInputStream);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
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
