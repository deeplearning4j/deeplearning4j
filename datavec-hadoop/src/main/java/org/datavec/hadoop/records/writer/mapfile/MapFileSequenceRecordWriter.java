/*
 *  * Copyright 2017 Skymind, Inc.
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

package org.datavec.hadoop.records.writer.mapfile;

import lombok.NonNull;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.io.LongWritable;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.SequenceRecordWriter;
import org.datavec.api.writable.*;
import org.datavec.api.writable.Writable;
import org.datavec.hadoop.records.reader.mapfile.record.RecordWritable;
import org.datavec.hadoop.records.reader.mapfile.record.SequenceRecordWritable;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Created by Alex on 07/07/2017.
 */
public class MapFileSequenceRecordWriter extends AbstractMapFileWriter<List<List<Writable>>> implements SequenceRecordWriter {


    public MapFileSequenceRecordWriter(File outputDir) {
        super(outputDir);
    }

    public MapFileSequenceRecordWriter(@NonNull File outputDir, int mapFileSplitSize){
        this(outputDir, mapFileSplitSize, null);
    }

    public MapFileSequenceRecordWriter(@NonNull File outputDir, WritableType convertTextTo) {
        this(outputDir, DEFAULT_MAP_FILE_SPLIT_SIZE, convertTextTo);
    }

    public MapFileSequenceRecordWriter(@NonNull File outputDir, int mapFileSplitSize, WritableType convertTextTo) {
        super(outputDir, mapFileSplitSize, convertTextTo);
    }

    @Override
    protected Class<? extends org.apache.hadoop.io.Writable> getValueClass() {
        return SequenceRecordWritable.class;
    }

    @Override
    protected org.apache.hadoop.io.Writable getHadoopWritable(List<List<Writable>> input) {
        if(convertTextTo != null){
            List<List<Writable>> newSeq = new ArrayList<>(input.size());
            for(List<Writable> l : input){
                newSeq.add(convertTextWritables(l));
            }
            input = newSeq;
        }

        return new SequenceRecordWritable(input);
    }
}
