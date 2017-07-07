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

import lombok.AllArgsConstructor;
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

    public static final int DEFAULT_NUM_OUTPUT_FILES = 1;
    public static final int DEFAULT_MAP_FILE_ROTATION_FREQ = 64;

    private final File outputDir;
    private final int numOutputFiles;
    private final int mapFileRotationFreq;
    private final WritableType convertTextTo;

    private final AtomicLong counter = new AtomicLong();
    private final AtomicBoolean isClosed = new AtomicBoolean();

    private File[] outputFiles;
    private MapFile.Writer[] writers;

    private List<Writable> tempList;

    public MapFileRecordWriter(File outputDir){
        this(outputDir, DEFAULT_NUM_OUTPUT_FILES);
    }

    public MapFileRecordWriter(@NonNull File outputDir, int numOutputFiles) {
        this(outputDir, numOutputFiles, DEFAULT_MAP_FILE_ROTATION_FREQ, null);
    }

    public MapFileRecordWriter(@NonNull File outputDir, int numOutputFiles, int mapFileRotationFreq, WritableType convertTextTo) {
        if(numOutputFiles <= 0){
            throw new IllegalArgumentException("Number of output files must be >= 1. Got: " + numOutputFiles);
        }
        this.outputDir = outputDir;
        this.numOutputFiles = numOutputFiles;
        this.mapFileRotationFreq = mapFileRotationFreq;
        if (convertTextTo == WritableType.Text) {
            convertTextTo = null;
        }
        this.convertTextTo = convertTextTo;

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

        if (outputFiles == null) {
            //Initialize
            SequenceFile.Writer.Option[] opts = new SequenceFile.Writer.Option[]{MapFile.Writer.keyClass(KEY_CLASS),
                    SequenceFile.Writer.valueClass(VALUE_CLASS)};

            org.apache.hadoop.conf.Configuration c = new org.apache.hadoop.conf.Configuration();

            this.outputFiles = new File[numOutputFiles];
            this.writers = new MapFile.Writer[numOutputFiles];
            for (int i = 0; i < numOutputFiles; i++) {
                String filename = String.format(DEFAULT_FILENAME_PATTERN, i);
                outputFiles[i] = new File(outputDir, filename);
                writers[i] = new MapFile.Writer(c, new Path(outputFiles[i].getAbsolutePath()), opts);
            }
        }

        long key = counter.getAndIncrement();
        MapFile.Writer w;
        if (numOutputFiles == 1) {
            w = writers[0];
        } else {
            w = writers[(int) ((key / mapFileRotationFreq) % numOutputFiles)];
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
