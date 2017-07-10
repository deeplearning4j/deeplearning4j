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
import org.apache.hadoop.io.MapFile;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.WritableComparable;
import org.datavec.api.conf.Configuration;
import org.datavec.api.writable.*;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * An abstract class For creating Hadoop map files, that underlies {@link MapFileRecordWriter} and
 * {@link MapFileSequenceRecordWriter}.
 *
 * @author Alex Black
 */
public abstract class AbstractMapFileWriter<T> {

    public static final String DEFAULT_FILENAME_PATTERN = "part-r-%1$05d";
    public static final Class<? extends WritableComparable> KEY_CLASS = org.apache.hadoop.io.LongWritable.class;

    /**
     * Configuration key for the map file interval.
     * This is defined in MapFile.Writer.INDEX_INTERVAL but unfortunately that field is private, hence cannot be
     * referenced here.
     */
    public static final String MAP_FILE_INDEX_INTERVAL_KEY = "io.map.index.interval";

    public static final int DEFAULT_MAP_FILE_SPLIT_SIZE = -1;
    public static final int DEFAULT_INDEX_INTERVAL = 1;

    protected final File outputDir;
    protected final int mapFileSplitSize;
    protected final WritableType convertTextTo;
    protected final int indexInterval;
    protected final String filenamePattern;
    protected org.apache.hadoop.conf.Configuration hadoopConfiguration;

    protected final AtomicLong counter = new AtomicLong();
    protected final AtomicBoolean isClosed = new AtomicBoolean();

    protected List<File> outputFiles = new ArrayList<>();
    protected List<MapFile.Writer> writers = new ArrayList<>();



    protected SequenceFile.Writer.Option[] opts;


    public AbstractMapFileWriter(File outputDir) {
        this(outputDir, DEFAULT_MAP_FILE_SPLIT_SIZE);
    }

    public AbstractMapFileWriter(@NonNull File outputDir, int mapFileSplitSize) {
        this(outputDir, mapFileSplitSize, null);
    }

    public AbstractMapFileWriter(@NonNull File outputDir, WritableType convertTextTo) {
        this(outputDir, DEFAULT_MAP_FILE_SPLIT_SIZE, convertTextTo);
    }

    public AbstractMapFileWriter(@NonNull File outputDir, int mapFileSplitSize, WritableType convertTextTo) {
        this(outputDir, mapFileSplitSize, convertTextTo, DEFAULT_INDEX_INTERVAL, new org.apache.hadoop.conf.Configuration());
    }

    public AbstractMapFileWriter(@NonNull File outputDir, int mapFileSplitSize, WritableType convertTextTo,
                                 int indexInterval, org.apache.hadoop.conf.Configuration hadoopConfiguration) {
        this(outputDir, mapFileSplitSize, convertTextTo, indexInterval, DEFAULT_FILENAME_PATTERN, hadoopConfiguration);
    }

    public AbstractMapFileWriter(@NonNull File outputDir, int mapFileSplitSize, WritableType convertTextTo,
                                 int indexInterval, String filenamePattern,
                                 org.apache.hadoop.conf.Configuration hadoopConfiguration) {
        if(indexInterval <= 0){
            throw new UnsupportedOperationException("Index interval: must be >= 0 (got: " + indexInterval + ")");
        }
        this.outputDir = outputDir;
        this.mapFileSplitSize = mapFileSplitSize;
        if (convertTextTo == WritableType.Text) {
            convertTextTo = null;
        }
        this.convertTextTo = convertTextTo;
        this.indexInterval = indexInterval;
        this.filenamePattern = filenamePattern;

        this.hadoopConfiguration = hadoopConfiguration;
        if(this.hadoopConfiguration.get(MAP_FILE_INDEX_INTERVAL_KEY) != null){
            this.hadoopConfiguration.set(MAP_FILE_INDEX_INTERVAL_KEY, String.valueOf(indexInterval));
        }

        opts = new SequenceFile.Writer.Option[]{MapFile.Writer.keyClass(KEY_CLASS),
                SequenceFile.Writer.valueClass(getValueClass())};

    }

    protected abstract Class<? extends org.apache.hadoop.io.Writable> getValueClass();


    public void setConf(Configuration conf) {

    }


    public Configuration getConf() {
        return null;
    }

    protected abstract org.apache.hadoop.io.Writable getHadoopWritable(T input);

    protected List<Writable> convertTextWritables(List<Writable> record) {
        List<Writable> newList;
        if (convertTextTo != null) {
            newList = new ArrayList<>(record.size());
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

        return newList;
    }

    public void write(T record) throws IOException {
        if (isClosed.get()) {
            throw new UnsupportedOperationException("Cannot write to MapFileRecordReader that has already been closed");
        }

        if (counter.get() == 0) {
            //Initialize first writer
            String filename = String.format(DEFAULT_FILENAME_PATTERN, 0);
            outputFiles.add(new File(outputDir, filename));
            writers.add(new MapFile.Writer(hadoopConfiguration, new Path(outputFiles.get(0).getAbsolutePath()), opts));
        }

        long key = counter.getAndIncrement();
        MapFile.Writer w;
        if (mapFileSplitSize <= 0) {
            w = writers.get(0);
        } else {
            int splitIdx = (int) (key / mapFileSplitSize);
            if (writers.size() <= splitIdx) {
                //Initialize new writer - next split
                String filename = String.format(DEFAULT_FILENAME_PATTERN, splitIdx);
                outputFiles.add(new File(outputDir, filename));
                writers.add(new MapFile.Writer(hadoopConfiguration, new Path(outputFiles.get(splitIdx).getAbsolutePath()), opts));
            }
            w = writers.get(splitIdx);
        }

        org.apache.hadoop.io.Writable hadoopWritable = getHadoopWritable(record);

        w.append(new org.apache.hadoop.io.LongWritable(key), hadoopWritable);
    }


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
