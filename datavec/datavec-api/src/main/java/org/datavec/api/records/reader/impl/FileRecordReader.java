/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

import lombok.Getter;
import lombok.Setter;
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
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * File reader/writer
 *
 * @author Adam Gibson
 */
public class FileRecordReader extends BaseRecordReader {

    protected Iterator<URI> locationsIterator;
    protected Configuration conf;
    protected URI currentUri;
    protected List<String> labels;
    protected boolean appendLabel = false;
    @Getter @Setter
    protected String charset = StandardCharsets.UTF_8.name(); //Using String as StandardCharsets.UTF_8 is not serializable

    public FileRecordReader() {}

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        super.initialize(split);
        doInitialize(split);
    }


    protected void doInitialize(InputSplit split) {

        if (labels == null && appendLabel) {
            URI[] locations = split.locations();
            if (locations.length > 0) {
                Set<String> labels = new HashSet<>();
                for(URI u : locations){
                    String[] pathSplit = u.toString().split("[/\\\\]");
                    labels.add(pathSplit[pathSplit.length-2]);
                }
                this.labels = new ArrayList<>(labels);
                Collections.sort(this.labels);
            }
        }
        locationsIterator = split.locationsIterator();
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

    private List<Writable> loadFromStream(URI uri, InputStream next, Charset charset) {
        List<Writable> ret = new ArrayList<>();
        try {
            if(!(next instanceof BufferedInputStream)){
                next = new BufferedInputStream(next);
            }
            String s = org.apache.commons.io.IOUtils.toString(next, charset);
            ret.add(new Text(s));
            if (appendLabel) {
                int idx = getLabel(uri);
                ret.add(new IntWritable(idx));
            }
        } catch (IOException e) {
            throw new IllegalStateException("Error reading from input stream: " + uri);
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
        return getLabel(currentUri);
    }

    public int getLabel(URI uri){
        String s = uri.toString();
        int lastIdx = Math.max(s.lastIndexOf('/'), s.lastIndexOf('\\'));    //Note: if neither are found, -1 is fine here
        String sub = s.substring(0, lastIdx);
        int secondLastIdx = Math.max(sub.lastIndexOf('/'), sub.lastIndexOf('\\'));
        String name = s.substring(secondLastIdx+1, lastIdx);
        return labels.indexOf(name);
    }

    public List<String> getLabels() {
        return labels;
    }

    public void setLabels(List<String> labels) {
        this.labels = labels;
    }

    @Override
    public boolean hasNext() {
        return locationsIterator.hasNext();
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
        URI next = locationsIterator.next();
        invokeListeners(next);

        List<Writable> ret;
        try(InputStream s = streamCreatorFn.apply(next)) {
            ret = loadFromStream(next, s, Charset.forName(charset));
        } catch (IOException e){
            throw new RuntimeException("Error reading from stream for URI: " + next);
        }

        return new org.datavec.api.records.impl.Record(ret,new RecordMetaDataURI(next, FileRecordReader.class));
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

            List<Writable> list;
            try(InputStream s = streamCreatorFn.apply(uri)) {
                list = loadFromStream(uri, s, Charset.forName(charset));
            } catch (IOException e){
                throw new RuntimeException("Error reading from stream for URI: " + uri);
            }

            out.add(new org.datavec.api.records.impl.Record(list, meta));
        }

        return out;
    }
}
