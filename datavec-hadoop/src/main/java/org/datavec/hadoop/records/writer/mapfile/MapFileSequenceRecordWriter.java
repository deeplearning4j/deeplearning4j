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
import org.datavec.api.records.writer.SequenceRecordWriter;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.WritableType;
import org.datavec.hadoop.records.reader.mapfile.record.SequenceRecordWritable;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * MapFileSequenceRecordWriter is used to write sequence values to a Hadoop MapFile, that can then be read by:
 * {@link org.datavec.hadoop.records.reader.mapfile.MapFileSequenceRecordReader}
 *
 * @author Alex Black
 * @see org.datavec.hadoop.records.reader.mapfile.MapFileSequenceRecordReader
 */
public class MapFileSequenceRecordWriter extends AbstractMapFileWriter<List<List<Writable>>> implements SequenceRecordWriter {

    /**
     * Constructor for all default values. Single output MapFile, no text writable conversion, default index
     * interval (1), default naming pattern.
     *
     * @param outputDir           Output directory for the map file(s)
     */
    public MapFileSequenceRecordWriter(File outputDir) {
        super(outputDir);
    }

    /**
     *
     * Constructor for most default values. Specified number of output MapFile s, no text writable conversion, default
     * index interval (1), default naming pattern.
     *
     * @param outputDir           Output directory for the map file(s)
     * @param mapFileSplitSize    Split size for the map file: if 0, use a single map file for all output. If > 0,
     *                            multiple map files will be used: each will contain a maximum of mapFileSplitSize
     *                            examples. This can be used to avoid having a single multi gigabyte map file, which may
     *                            be undesirable in some cases (transfer across the network, for example).
     */
    public MapFileSequenceRecordWriter(@NonNull File outputDir, int mapFileSplitSize){
        this(outputDir, mapFileSplitSize, null);
    }

    /**
     *
     * @param outputDir           Output directory for the map file(s)
     * @param convertTextTo       If null: Make no changes to Text writable objects. If non-null, Text writable instances
     *                            will be converted to this type. This is useful, when would rather store numerical values
     *                            even if the original record reader produces strings/text.
     */
    public MapFileSequenceRecordWriter(@NonNull File outputDir, WritableType convertTextTo) {
        this(outputDir, DEFAULT_MAP_FILE_SPLIT_SIZE, convertTextTo);
    }

    /**
     *
     * @param outputDir           Output directory for the map file(s)
     * @param mapFileSplitSize    Split size for the map file: if 0, use a single map file for all output. If > 0,
     *                            multiple map files will be used: each will contain a maximum of mapFileSplitSize
     *                            examples. This can be used to avoid having a single multi gigabyte map file, which may
     *                            be undesirable in some cases (transfer across the network, for example).
     * @param convertTextTo       If null: Make no changes to Text writable objects. If non-null, Text writable instances
     *                            will be converted to this type. This is useful, when would rather store numerical values
     *                            even if the original record reader produces strings/text.
     */
    public MapFileSequenceRecordWriter(@NonNull File outputDir, int mapFileSplitSize, WritableType convertTextTo) {
        super(outputDir, mapFileSplitSize, convertTextTo);
    }

    /**
     *
     * @param outputDir           Output directory for the map file(s)
     * @param mapFileSplitSize    Split size for the map file: if 0, use a single map file for all output. If > 0,
     *                            multiple map files will be used: each will contain a maximum of mapFileSplitSize
     *                            examples. This can be used to avoid having a single multi gigabyte map file, which may
     *                            be undesirable in some cases (transfer across the network, for example).
     * @param convertTextTo       If null: Make no changes to Text writable objects. If non-null, Text writable instances
     *                            will be converted to this type. This is useful, when would rather store numerical values
     *                            even if the original record reader produces strings/text.
     * @param hadoopConfiguration Hadoop configuration.
     */
    public MapFileSequenceRecordWriter(@NonNull File outputDir, int mapFileSplitSize, WritableType convertTextTo,
                                       org.apache.hadoop.conf.Configuration hadoopConfiguration) {
        super(outputDir, mapFileSplitSize, convertTextTo, DEFAULT_INDEX_INTERVAL, hadoopConfiguration);
    }

    /**
     *
     * @param outputDir           Output directory for the map file(s)
     * @param mapFileSplitSize    Split size for the map file: if 0, use a single map file for all output. If > 0,
     *                            multiple map files will be used: each will contain a maximum of mapFileSplitSize
     *                            examples. This can be used to avoid having a single multi gigabyte map file, which may
     *                            be undesirable in some cases (transfer across the network, for example).
     * @param convertTextTo       If null: Make no changes to Text writable objects. If non-null, Text writable instances
     *                            will be converted to this type. This is useful, when would rather store numerical values
     *                            even if the original record reader produces strings/text.
     * @param indexInterval       Index interval for the Map file. Defaults to 1, which is suitable for most cases
     * @param hadoopConfiguration Hadoop configuration.
     */
    public MapFileSequenceRecordWriter(@NonNull File outputDir, int mapFileSplitSize, WritableType convertTextTo,
                                       int indexInterval, org.apache.hadoop.conf.Configuration hadoopConfiguration) {
        super(outputDir, mapFileSplitSize, convertTextTo, indexInterval, hadoopConfiguration);
    }

    /**
     *
     * @param outputDir           Output directory for the map file(s)
     * @param mapFileSplitSize    Split size for the map file: if 0, use a single map file for all output. If > 0,
     *                            multiple map files will be used: each will contain a maximum of mapFileSplitSize
     *                            examples. This can be used to avoid having a single multi gigabyte map file, which may
     *                            be undesirable in some cases (transfer across the network, for example).
     * @param convertTextTo       If null: Make no changes to Text writable objects. If non-null, Text writable instances
     *                            will be converted to this type. This is useful, when would rather store numerical values
     *                            even if the original record reader produces strings/text.
     * @param indexInterval       Index interval for the Map file. Defaults to 1, which is suitable for most cases
     * @param filenamePattern     The naming pattern for the map files. Used with String.format(pattern, int)
     * @param hadoopConfiguration Hadoop configuration.
     */
    public MapFileSequenceRecordWriter(@NonNull File outputDir, int mapFileSplitSize, WritableType convertTextTo,
                                       int indexInterval, String filenamePattern,
                                       org.apache.hadoop.conf.Configuration hadoopConfiguration) {
        super(outputDir, mapFileSplitSize, convertTextTo, indexInterval, filenamePattern, hadoopConfiguration);
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
