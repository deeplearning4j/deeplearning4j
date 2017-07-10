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
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.WritableType;
import org.datavec.hadoop.records.reader.mapfile.record.RecordWritable;

import java.io.File;
import java.util.List;

/**
 * Created by Alex on 07/07/2017.
 */
public class MapFileRecordWriter extends AbstractMapFileWriter<List<Writable>> implements RecordWriter {


    public MapFileRecordWriter(File outputDir) {
        super(outputDir);
    }

    public MapFileRecordWriter(@NonNull File outputDir, int mapFileSplitSize){
        this(outputDir, mapFileSplitSize, null);
    }

    public MapFileRecordWriter(@NonNull File outputDir, WritableType convertTextTo) {
        this(outputDir, DEFAULT_MAP_FILE_SPLIT_SIZE, convertTextTo);
    }

    public MapFileRecordWriter(@NonNull File outputDir, int mapFileSplitSize, WritableType convertTextTo) {
        super(outputDir, mapFileSplitSize, convertTextTo);
    }

    public MapFileRecordWriter(@NonNull File outputDir, int mapFileSplitSize, WritableType convertTextTo,
                                       org.apache.hadoop.conf.Configuration hadoopConfiguration) {
        super(outputDir, mapFileSplitSize, convertTextTo, DEFAULT_INDEX_INTERVAL, hadoopConfiguration);
    }

    public MapFileRecordWriter(@NonNull File outputDir, int mapFileSplitSize, WritableType convertTextTo,
                               int indexInterval, org.apache.hadoop.conf.Configuration hadoopConfiguration) {
        super(outputDir, mapFileSplitSize, convertTextTo, indexInterval, hadoopConfiguration);
    }

    public MapFileRecordWriter(@NonNull File outputDir, int mapFileSplitSize, WritableType convertTextTo,
                                       int indexInterval, String filenamePattern,
                                       org.apache.hadoop.conf.Configuration hadoopConfiguration) {
        super(outputDir, mapFileSplitSize, convertTextTo, indexInterval, filenamePattern, hadoopConfiguration);
    }

    @Override
    protected Class<? extends org.apache.hadoop.io.Writable> getValueClass() {
        return RecordWritable.class;
    }

    @Override
    protected org.apache.hadoop.io.Writable getHadoopWritable(List<Writable> input) {
        if(convertTextTo != null){
            input = convertTextWritables(input);
        }

        return new RecordWritable(input);
    }
}
