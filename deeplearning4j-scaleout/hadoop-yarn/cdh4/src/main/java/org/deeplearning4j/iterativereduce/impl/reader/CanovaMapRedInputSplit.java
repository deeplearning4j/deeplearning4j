/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.iterativereduce.impl.reader;

import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.InputSplit;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.net.URI;

/**
 * Canova input split: canova ---> hadoop
 *
 * @author Adam Gibson
 */
public class CanovaMapRedInputSplit implements org.canova.api.split.InputSplit {

    private InputSplit split;
    private org.apache.hadoop.mapreduce.InputSplit inputSplit;
    private URI[] uris;

    public CanovaMapRedInputSplit(InputSplit split) {
        this.split = split;
        try {
            FileSplit split2 = (FileSplit) split;
            //create from the path
            uris = new URI[1];
            uris[0] = URI.create(split2.getPath().toString());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }

    public CanovaMapRedInputSplit(org.apache.hadoop.mapreduce.InputSplit inputSplit) {
        this.inputSplit = inputSplit;
        try {
            FileSplit split2 = (FileSplit) split;
            //create from the path
            uris = new URI[1];
            uris[0] = URI.create(split2.getPath().toString());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }

    @Override
    public long length() {
        try {
            return split.getLength();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public URI[] locations() {
        return uris;

    }

    @Override
    public void write(DataOutput out) throws IOException {
    }

    @Override
    public void readFields(DataInput in) throws IOException {

    }

    @Override
    public double toDouble(){
        throw new UnsupportedOperationException();
    }

    @Override
    public float toFloat(){
        throw new UnsupportedOperationException();
    }

    @Override
    public int toInt(){
        throw new UnsupportedOperationException();
    }

    @Override
    public long toLong(){
        throw new UnsupportedOperationException();
    }
}
