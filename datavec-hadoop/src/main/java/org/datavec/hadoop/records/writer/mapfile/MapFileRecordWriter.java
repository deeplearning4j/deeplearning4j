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
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.MapFile;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.WritableComparable;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.writable.*;
import org.datavec.hadoop.records.reader.mapfile.record.RecordWritable;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Created by Alex on 07/07/2017.
 */
public class MapFileRecordWriter implements RecordWriter {

    public static final String DEFAULT_FILENAME_PATTERN = "part-r-%1$05d";
    public static final Class<? extends WritableComparable> KEY_CLASS = LongWritable.class;
    public static final Class<? extends org.apache.hadoop.io.Writable> VALUE_CLASS = RecordWritable.class;

    public static final int DEFAULT_MAP_FILE_SPLIT_SIZE = -1;

    private final File outputDir;
    private final int mapFileSplitSize;
    private final WritableType convertTextTo;

    private final AtomicLong counter = new AtomicLong();
    private final AtomicBoolean isClosed = new AtomicBoolean();

    private List<File> outputFiles = new ArrayList<>();
    private List<MapFile.Writer> writers = new ArrayList<>();

    org.apache.hadoop.conf.Configuration c = new org.apache.hadoop.conf.Configuration();

    private List<Writable> tempList;

    private SequenceFile.Writer.Option[] opts;


    public MapFileRecordWriter(File outputDir){
        this(outputDir, DEFAULT_MAP_FILE_SPLIT_SIZE);
    }

    public MapFileRecordWriter(@NonNull File outputDir, int mapFileSplitSize){
        this(outputDir, mapFileSplitSize, null);
    }

    public MapFileRecordWriter(@NonNull File outputDir, WritableType convertTextTo) {
        this(outputDir, DEFAULT_MAP_FILE_SPLIT_SIZE, convertTextTo);
    }

    public MapFileRecordWriter(@NonNull File outputDir, int mapFileSplitSize, WritableType convertTextTo) {
        this.outputDir = outputDir;
        this.mapFileSplitSize = mapFileSplitSize;
        if (convertTextTo == WritableType.Text) {
            convertTextTo = null;
        }
        this.convertTextTo = convertTextTo;

        opts = new SequenceFile.Writer.Option[]{MapFile.Writer.keyClass(KEY_CLASS),
                SequenceFile.Writer.valueClass(VALUE_CLASS)};

    }


    @Override
    public void setConf(Configuration conf) {

    }

    @Override
    public Configuration getConf() {
        return null;
    }

    @Override
    public void write(List<Writable> record) throws IOException {
        if (isClosed.get()) {
            throw new UnsupportedOperationException("Cannot write to MapFileRecordReader that has already been closed");
        }

        if (counter.get() == 0) {
            //Initialize first writer
            String filename = String.format(DEFAULT_FILENAME_PATTERN, 0);
            outputFiles.add(new File(outputDir, filename));
            writers.add(new MapFile.Writer(c, new Path(outputFiles.get(0).getAbsolutePath()), opts));
        }

        long key = counter.getAndIncrement();
        MapFile.Writer w;
        if (mapFileSplitSize <= 0) {
            w = writers.get(0);
        } else {
            int splitIdx = (int)(key / mapFileSplitSize);
            if(writers.size() <= splitIdx){
                //Initialize new writer - next split
                String filename = String.format(DEFAULT_FILENAME_PATTERN, splitIdx);
                outputFiles.add(new File(outputDir, filename));
                writers.add(new MapFile.Writer(c, new Path(outputFiles.get(splitIdx).getAbsolutePath()), opts));
            }
            w = writers.get(splitIdx);
        }

        List<Writable> newList;
        if (convertTextTo != null) {
            if (tempList == null) {
                tempList = new ArrayList<>(record.size());
            }
            newList = tempList;
            for (Writable writable : record) {
                Writable newWritable;
                if (writable.getType() == WritableType.Text) {
                    switch (convertTextTo) {
                        case Byte:
                            newWritable = new ByteWritable((byte) writable.toInt());
                            break;
                        case Double:
                            newWritable = new DoubleWritable(writable.toDouble());
                            break;
                        case Float:
                            newWritable = new FloatWritable(writable.toFloat());
                            break;
                        case Int:
                            newWritable = new IntWritable(writable.toInt());
                            break;
                        case Long:
                            newWritable = new org.datavec.api.writable.LongWritable(writable.toLong());
                            break;
                        default:
                            throw new UnsupportedOperationException("Cannot convert text to: " + convertTextTo);
                    }
                } else {
                    newWritable = writable;
                }
                newList.add(newWritable);
            }
        } else {
            newList = record;
        }

        RecordWritable rw = new RecordWritable(newList);

        w.append(new LongWritable(key), rw);

        if (tempList != null) {
            tempList.clear();
        }
    }

    @Override
    public void close() {
        try {
            for (MapFile.Writer w : writers) {
                w.close();
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            isClosed.set(true);
        }
    }
}
