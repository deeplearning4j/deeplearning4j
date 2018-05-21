/*-
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

package org.datavec.hadoop.records.reader.mapfile;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.MapFile;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.util.ReflectionUtils;
import org.nd4j.linalg.primitives.Pair;
import org.datavec.hadoop.records.reader.mapfile.index.LongIndexToKey;
import org.datavec.hadoop.records.reader.mapfile.record.RecordWritable;

import java.io.Closeable;
import java.io.IOException;
import java.util.Collections;
import java.util.List;

/**
 * A wrapper around a Hadoop {@link MapFile.Reader}, used in {@link MapFileRecordReader} and {@link MapFileSequenceRecordReader}
 *
 * <b>Note</b>: This also handles multiple map files, such as the output from Spark, which gives a set of map files
 * in directories like /part-r-00000, /part-r-00001
 *
 * @author Alex Black
 */
public class MapFileReader<V> implements Closeable {

    private MapFile.Reader[] readers;
    private IndexToKey indexToKey;
    private Class<? extends Writable> recordClass;
    private List<Pair<Long, Long>> recordIndexesEachReader;
    private Long numRecords;


    public MapFileReader(String path) throws Exception {
        this(path, new LongIndexToKey(), RecordWritable.class);
    }

    /**
     * @param path        Path (directory) of the MapFile
     * @param indexToKey  Instance used to convert long indices to key values. This allows for lookup by key
     * @param recordClass Class of the records in the MapFile
     * @throws IOException If an error occurs during opening or initialisation
     */
    public MapFileReader(String path, IndexToKey indexToKey, Class<? extends Writable> recordClass) throws IOException {
        this(Collections.singletonList(path), indexToKey, recordClass);
    }

    public MapFileReader(List<String> paths, IndexToKey indexToKey, Class<? extends Writable> recordClass)
                    throws IOException {

        this.indexToKey = indexToKey;
        this.recordClass = recordClass;
        this.readers = new MapFile.Reader[paths.size()];

        SequenceFile.Reader.Option[] opts = new SequenceFile.Reader.Option[0];

        Configuration config = new Configuration();
        for (int i = 0; i < paths.size(); i++) {
            readers[i] = new MapFile.Reader(new Path(paths.get(i)), config, opts);
            if (readers[i].getValueClass() != recordClass) {
                throw new UnsupportedOperationException("MapFile record class: " + readers[i].getValueClass()
                                + ", but got class " + recordClass + ", path = " + paths.get(i));
            }
        }

        recordIndexesEachReader = indexToKey.initialize(readers, recordClass);
    }

    /**
     * Determine the total number of records in the map file, using the {@link IndexToKey} instance
     *
     * @return  Total number of records and the map file
     */
    public long numRecords() {
        if (numRecords == null) {
            try {
                numRecords = indexToKey.getNumRecords();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        return numRecords;
    }

    /**
     * It a single record from the map file for the given index
     *
     * @param index Index, between 0 and numRecords()-1
     * @return Value from the MapFile
     * @throws IOException If an error occurs during reading
     */
    public V getRecord(long index) throws IOException {
        //First: determine which reader to read from...
        int readerIdx = -1;
        for (int i = 0; i < recordIndexesEachReader.size(); i++) {
            Pair<Long, Long> p = recordIndexesEachReader.get(i);
            if (index >= p.getFirst() && index <= p.getSecond()) {
                readerIdx = i;
                break;
            }
        }
        if (readerIdx == -1) {
            throw new IllegalStateException("Index not found in any reader: " + index);
        }

        WritableComparable key = indexToKey.getKeyForIndex(index);
        Writable value = ReflectionUtils.newInstance(recordClass, null);

        V v = (V) readers[readerIdx].get(key, value);
        return v;
    }


    @Override
    public void close() throws IOException {
        for (MapFile.Reader r : readers) {
            r.close();
        }
    }
}
