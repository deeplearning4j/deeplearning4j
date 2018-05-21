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

package org.datavec.api.records.reader.impl;

import org.apache.commons.io.FileUtils;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

import java.io.*;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * File reader/writer
 *
 * @author Adam Gibson
 */
public class FileRecordReader extends BaseRecordReader {

    protected Iterator<File> iter;
    protected Iterator<String> locationsIterator;
    protected Configuration conf;
    protected File currentFile;
    protected List<String> labels;
    protected boolean appendLabel = false;
    protected InputSplit inputSplit;

    public FileRecordReader() {}

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        doInitialize(split);
        this.inputSplit = split;
    }


    protected void doInitialize(InputSplit split) {

        if (labels == null && appendLabel) {
            URI[] locations = split.locations();
            if (locations.length > 0) {
                //root dir relative to example where the label is the parent directory and the root directory is
                //recursively the parent of that
                File parent = new File(locations[0]).getParentFile().getParentFile();
                //calculate the labels relative to the parent file
                labels = new ArrayList<>();

                for (File labelDir : parent.listFiles())
                    labels.add(labelDir.getName());
            }
        }
        locationsIterator = split.locationsPathIterator();
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        appendLabel = conf.getBoolean(APPEND_LABEL, true);
        doInitialize(split);
        this.inputSplit = split;
        this.conf = conf;
    }

    @Override
    public List<Writable> next() {
        return nextRecord().getRecord();
    }

    private List<Writable> loadFromFile(File next) {
        List<Writable> ret = new ArrayList<>();
        try {
            ret.add(new Text(FileUtils.readFileToString(next)));
            if (appendLabel)
                ret.add(new IntWritable(labels.indexOf(next.getParentFile().getName())));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ret;
    }

    /**
     * Return the current label.
     * The index of the current file's parent directory
     * in the label list
     * @return The index of the current file's parent directory
     */
    public int getCurrentLabel() {
        return labels.indexOf(currentFile.getParentFile().getName());
    }

    public List<String> getLabels() {
        return labels;
    }

    public void setLabels(List<String> labels) {
        this.labels = labels;
    }

    @Override
    public boolean hasNext() {
        if (iter != null && iter.hasNext()) {
            return true;
        }
        if (!locationsIterator.hasNext()) {
            return false;
        }
        // iter is exhausted, set to iterate of the next location
        this.advanceToNextLocation();
        return iter != null && iter.hasNext();
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
    public List<List<Writable>> next(int num) {
        List<List<Writable>> ret = new ArrayList<>(num);
        int numBatches = 0;
        while (hasNext() && numBatches < num) {
            ret.add(next());
        }

        return ret;
    }
    @Override
    public void reset() {
        if (inputSplit == null)
            throw new UnsupportedOperationException("Cannot reset without first initializing");
        try {
            doInitialize(inputSplit);
        } catch (Exception e) {
            throw new RuntimeException("Error during LineRecordReader reset", e);
        }
    }

    @Override
    public boolean resetSupported() {
        if(inputSplit != null){
            return inputSplit.resetSupported();
        }
        return false;   //reset() throws exception on reset() if inputSplit is null
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        invokeListeners(uri);
        //Here: reading the entire file to a Text writable
        BufferedReader br = new BufferedReader(new InputStreamReader(dataInputStream));
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = br.readLine()) != null) {
            sb.append(line).append("\n");
        }
        return Collections.singletonList((Writable) new Text(sb.toString()));
    }

    @Override
    public Record nextRecord() {
        if (iter == null || !iter.hasNext()) {
            this.advanceToNextLocation();
        }
        File next = iter.next();
        this.currentFile = next;
        invokeListeners(next);
        List<Writable> ret = loadFromFile(next);

        return new org.datavec.api.records.impl.Record(ret,
                new RecordMetaDataURI(next.toURI(), FileRecordReader.class));
    }

    protected File nextFile() {
        if (iter == null || !iter.hasNext()) {
            this.advanceToNextLocation();
        }
        File next = iter.next();
        this.currentFile = next;
        return next;
    }

    protected void advanceToNextLocation () {
        //File file;
        String path = locationsIterator.next(); // should always have file:// preceding
        if(!path.startsWith("file:")){
            path = "file:///" + path;
        }
        if(path.contains("\\")){
            path = path.replaceAll("\\\\","/");
        }
        File file = new File(URI.create(path));
        if (file.isDirectory())
            iter = FileUtils.iterateFiles(file, null, true);
        else
            iter = Collections.singletonList(file).iterator();
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return loadFromMetaData(Collections.singletonList(recordMetaData)).get(0);
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        List<Record> out = new ArrayList<>();

        for (RecordMetaData meta : recordMetaDatas) {
            URI uri = meta.getURI();

            File f = new File(uri);
            List<Writable> list = loadFromFile(f);
            out.add(new org.datavec.api.records.impl.Record(list, meta));
        }

        return out;
    }
}
