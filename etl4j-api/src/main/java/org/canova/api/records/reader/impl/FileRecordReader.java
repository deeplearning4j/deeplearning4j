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

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.canova.api.conf.Configuration;
import org.canova.api.io.data.Text;
import org.canova.api.records.reader.BaseRecordReader;
import org.canova.api.split.InputSplit;
import org.canova.api.writable.Writable;


import java.io.*;
import java.net.URI;
import java.util.*;

/**
 * File reader/writer
 *
 * @author Adam Gibson
 */
public class FileRecordReader extends BaseRecordReader {

    protected Iterator<File> iter;
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
        URI[] locations = split.locations();

        if(locations != null && locations.length >= 1) {
            if(locations.length > 1) {
                List<File> allFiles = new ArrayList<>();
                for(URI location : locations) {
                    File iter = new File(location);
                    if(labels == null && appendLabel) {
                        //root dir relative to example where the label is the parent directory and the root directory is
                        //recursively the parent of that
                        File parent = iter.getParentFile().getParentFile();
                        //calculate the labels relative to the parent file
                        labels = new ArrayList<>();

                        for(File labelDir : parent.listFiles())
                            labels.add(labelDir.getName());
                    }

                    if(iter.isDirectory()) {
                        Iterator<File> allFiles2 = FileUtils.iterateFiles(iter,null,true);
                        while(allFiles2.hasNext())
                            allFiles.add(allFiles2.next());
                    }

                    else
                        allFiles.add(iter);
                }

                iter = allFiles.listIterator();
            }
            else {
                File curr = new File(locations[0]);
                if(curr.isDirectory())
                    iter = FileUtils.iterateFiles(curr,null,true);
                else
                    iter = Collections.singletonList(curr).iterator();
            }
        }

    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        appendLabel = conf.getBoolean(APPEND_LABEL,true);
        doInitialize(split);
        this.inputSplit = split;
        this.conf = conf;
    }

    @Override
    public Collection<Writable> next() {
        List<Writable> ret = new ArrayList<>();
        try {
            File next = iter.next();
            this.currentFile = next;
            invokeListeners(next);
            ret.add(new Text(FileUtils.readFileToString(next)));
        } catch (IOException e) {
            e.printStackTrace();
        }
        if(iter.hasNext()) {
            return ret;
        }
        else {
            if(iter.hasNext()) {
                try {
                    File next = iter.next();
                    this.currentFile = next;
                    invokeListeners(next);
                    ret.add(new Text(FileUtils.readFileToString(next)));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

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
    public void reset() {
        if(inputSplit == null) throw new UnsupportedOperationException("Cannot reset without first initializing");
        try{
            doInitialize(inputSplit);
        }catch(Exception e){
            throw new RuntimeException("Error during LineRecordReader reset",e);
        }
    }

    @Override
    public Collection<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        invokeListeners(uri);
        //Here: reading the entire file to a Text writable
        BufferedReader br = new BufferedReader(new InputStreamReader(dataInputStream));
        StringBuilder sb = new StringBuilder();
        String line;
        while((line = br.readLine()) != null){
            sb.append(line).append("\n");
        }
        return Collections.singletonList((Writable)new Text(sb.toString()));
    }

}
