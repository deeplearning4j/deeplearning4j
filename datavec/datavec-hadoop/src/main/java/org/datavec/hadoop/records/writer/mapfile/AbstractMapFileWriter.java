/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.datavec.hadoop.records.writer.mapfile;

import lombok.NonNull;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.MapFile;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.WritableComparable;
import org.datavec.api.conf.Configuration;
import org.datavec.api.split.partition.PartitionMetaData;
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


    /**
     * Constructor for all default values. Single output MapFile, no text writable conversion, default index
     * interval (1), default naming pattern.
     *
     * @param outputDir           Output directory for the map file(s)
     */
    public AbstractMapFileWriter(File outputDir) {
        this(outputDir, DEFAULT_MAP_FILE_SPLIT_SIZE);
    }

    /**
     *
     * Constructor for most default values. Specified number of output MapFile s, no text writable conversion, default
     * index interval (1), default naming pattern.
     *
     * @param outputDir           Output directory for the map file(s)
     * @param mapFileSplitSize    Split size for the map file: if 0, use a single map file for all output. If > 0,
     *                            multiple map files will be used: each will contain a maximum of mapFileSplitSize.
     *                            This can be used to avoid having a single multi gigabyte map file, which may be
     *                            undesirable in some cases (transfer across the network, for example)
     */
    public AbstractMapFileWriter(@NonNull File outputDir, int mapFileSplitSize) {
        this(outputDir, mapFileSplitSize, null);
    }

    /**
     *
     * @param outputDir           Output directory for the map file(s)
     * @param convertTextTo       If null: Make no changes to Text writable objects. If non-null, Text writable instances
     *                            will be converted to this type. This is useful, when would rather store numerical values
     *                            even if the original record reader produces strings/text.
     */
    public AbstractMapFileWriter(@NonNull File outputDir, WritableType convertTextTo) {
        this(outputDir, DEFAULT_MAP_FILE_SPLIT_SIZE, convertTextTo);
    }

    /**
     *
     * @param outputDir           Output directory for the map file(s)
     * @param mapFileSplitSize    Split size for the map file: if 0, use a single map file for all output. If > 0,
     *                            multiple map files will be used: each will contain a maximum of mapFileSplitSize.
     *                            This can be used to avoid having a single multi gigabyte map file, which may be
     *                            undesirable in some cases (transfer across the network, for example)
     * @param convertTextTo       If null: Make no changes to Text writable objects. If non-null, Text writable instances
     *                            will be converted to this type. This is useful, when would rather store numerical values
     *                            even if the original record reader produces strings/text.
     */
    public AbstractMapFileWriter(@NonNull File outputDir, int mapFileSplitSize, WritableType convertTextTo) {
        this(outputDir, mapFileSplitSize, convertTextTo, DEFAULT_INDEX_INTERVAL, new org.apache.hadoop.conf.Configuration());
    }

    /**
     *
     * @param outputDir           Output directory for the map file(s)
     * @param mapFileSplitSize    Split size for the map file: if 0, use a single map file for all output. If > 0,
     *                            multiple map files will be used: each will contain a maximum of mapFileSplitSize.
     *                            This can be used to avoid having a single multi gigabyte map file, which may be
     *                            undesirable in some cases (transfer across the network, for example)
     * @param convertTextTo       If null: Make no changes to Text writable objects. If non-null, Text writable instances
     *                            will be converted to this type. This is useful, when would rather store numerical values
     *                            even if the original record reader produces strings/text.
     * @param indexInterval       Index interval for the Map file. Defaults to 1, which is suitable for most cases
     * @param hadoopConfiguration Hadoop configuration.
     */
    public AbstractMapFileWriter(@NonNull File outputDir, int mapFileSplitSize, WritableType convertTextTo,
                                 int indexInterval, org.apache.hadoop.conf.Configuration hadoopConfiguration) {
        this(outputDir, mapFileSplitSize, convertTextTo, indexInterval, DEFAULT_FILENAME_PATTERN, hadoopConfiguration);
    }

    /**
     *
     * @param outputDir           Output directory for the map file(s)
     * @param mapFileSplitSize    Split size for the map file: if 0, use a single map file for all output. If > 0,
     *                            multiple map files will be used: each will contain a maximum of mapFileSplitSize.
     *                            This can be used to avoid having a single multi gigabyte map file, which may be
     *                            undesirable in some cases (transfer across the network, for example)
     * @param convertTextTo       If null: Make no changes to Text writable objects. If non-null, Text writable instances
     *                            will be converted to this type. This is useful, when would rather store numerical values
     *                            even if the original record reader produces strings/text.
     * @param indexInterval       Index interval for the Map file. Defaults to 1, which is suitable for most cases
     * @param filenamePattern     The naming pattern for the map files. Used with String.format(pattern, int)
     * @param hadoopConfiguration Hadoop configuration.
     */
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

    public PartitionMetaData write(T record) throws IOException {
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

        return PartitionMetaData.builder().numRecordsUpdated(1).build();
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
