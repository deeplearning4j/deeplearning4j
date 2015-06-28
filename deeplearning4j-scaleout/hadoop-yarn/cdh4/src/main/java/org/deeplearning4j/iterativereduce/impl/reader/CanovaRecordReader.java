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

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;

/**
 * canova record reader
 * @author Adam Gibson
 */
public class CanovaRecordReader extends RecordReader<Long, Collection<Writable>> {
    private org.canova.api.records.reader.RecordReader recordReader;
    private int numRecords = 0;
    private Collection<Writable> currRecord;

    public CanovaRecordReader(org.canova.api.records.reader.RecordReader recordReader) {
        this.recordReader = recordReader;
    }

    public void initialize(InputSplit inputSplit) throws IOException, InterruptedException {
        recordReader.initialize(new CanovaMapRedInputSplit(inputSplit));
    }









    @Override
    public void close() throws IOException {
        recordReader.close();
    }

    @Override
    public void initialize(org.apache.hadoop.mapreduce.InputSplit inputSplit, TaskAttemptContext taskAttemptContext) throws IOException, InterruptedException {
        recordReader.initialize(new CanovaMapRedInputSplit(inputSplit));

    }

    @Override
    public boolean nextKeyValue() throws IOException, InterruptedException {
        return false;
    }

    @Override
    public Long getCurrentKey() throws IOException, InterruptedException {
        return null;
    }

    @Override
    public Collection<Writable> getCurrentValue() throws IOException, InterruptedException {
        if(recordReader.hasNext()) {
            Collection<org.canova.api.writable.Writable> writables = recordReader.next();
            Collection<Writable> wrapped = new ArrayList<>();
            for(org.canova.api.writable.Writable writable : writables)
                wrapped.add(new CanovaWritableDelegate(writable));
            currRecord = wrapped;


        }

        return currRecord;
    }

    @Override
    public float getProgress() throws IOException {
        return 0;
    }
}
