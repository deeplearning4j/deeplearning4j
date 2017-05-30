/*
 *  * Copyright 2016 Skymind, Inc.
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

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataIndex;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.RandomUtils;
import org.datavec.api.writable.Writable;
import org.datavec.hadoop.records.reader.mapfile.index.LongIndexToKey;
import org.datavec.hadoop.records.reader.mapfile.record.RecordWritable;

import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.*;

/**
 * A {@link RecordReader} implementation for reading from a Hadoop {@link org.apache.hadoop.io.MapFile}<br>
 * <p>
 * A typical use case is with {@link org.datavec.api.transform.TransformProcess} executed on Spark (perhaps Spark
 * local), followed by non-distributed training on a single machine. For example:
 * <pre>
 *  {@code
 *  JavaRDD<List<Writable>> myRDD = ...;
 *  String mapFilePath = ...;
 *  SparkStorageUtils.saveMapFile( mapFilePath, myRDD );
 *
 *  RecordReader rr = new MapFileRecordReader();
 *  rr.initialize( new FileSplit( new File( mapFilePath ) ) );
 *  //Pass to DataSetIterator or similar
 *  }
 * </pre>
 *
 * @author Alex Black
 */
public class MapFileRecordReader implements RecordReader {
    private static final Class<? extends org.apache.hadoop.io.Writable> recordClass = RecordWritable.class;

    private final IndexToKey indexToKey;
    private MapFileReader<RecordWritable> mapFileReader;
    private URI uri;
    private List<RecordListener> listeners;

    private long numRecords;
    private long position;
    private Random rng;
    private int[] order;

    /**
     * Create a MapFileRecordReader with no randomisation, and assuming MapFile keys are {@link org.apache.hadoop.io.LongWritable}
     * values
     */
    public MapFileRecordReader() throws Exception {
        this(new LongIndexToKey(), null);
    }

    /**
     * Create a MapFileRecordReader with optional randomisation, and assuming MapFile keys are
     * {@link org.apache.hadoop.io.LongWritable} values
     *
     * @param rng If non-null, will be used to randomize the order of examples
     *
     */
    public MapFileRecordReader(Random rng) {
        this(new LongIndexToKey(), rng);
    }

    /**
     * Create a MapFileRecordReader with optional randomisation, with a custom {@link IndexToKey} instance to
     * handle MapFile keys
     *
     * @param indexToKey Handles conversion between long indices and key values (see for example {@link LongIndexToKey}
     * @param rng If non-null, will be used to randomize the order of examples
     *
     */
    public MapFileRecordReader(IndexToKey indexToKey, Random rng) {
        this.indexToKey = indexToKey;
        this.rng = rng;
    }

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        initialize(null, split);
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        URI[] uris = split.locations();

        //Check URIs are correct: we expect /data and /index files...
        if (uris.length == 0) {
            throw new IllegalStateException("Cannot initialize MapFileSequenceRecordReader: could not find data and index files in input split");
        }

        uri = uris[0];
        File f = new File(uri);
        if (!f.isDirectory()) {
            f = f.getParentFile();
            uri = f.toURI();
        }

        File indexFile = new File(f, "index");
        File dataFile = new File(f, "data");

        if (!indexFile.exists()) {
            throw new IOException("Could not find index file at " + indexFile.getAbsolutePath() + " - must have MapFile "
                    + "index and data files at the input split location");
        }
        if (!dataFile.exists()) {
            throw new IOException("Could not find data file at " + dataFile.getAbsolutePath() + " - must have MapFile "
                    + "index and data files at the input split location");
        }


        if (mapFileReader != null) {
            mapFileReader.close();
            mapFileReader = null;
        }

        this.mapFileReader = new MapFileReader<>(uri.getPath(), indexToKey, recordClass);
        this.numRecords = mapFileReader.numRecords();

        if (rng != null) {
            order = new int[(int) numRecords];
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            RandomUtils.shuffleInPlace(order, rng);
        }
    }

    @Override
    public void setConf(Configuration conf) {

    }

    @Override
    public Configuration getConf() {
        return null;
    }

    @Override
    public boolean batchesSupported() {
        return false;
    }

    @Override
    public List<Writable> next(int num) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<Writable> next() {
        return next(false).getRecord();
    }

    @Override
    public boolean hasNext() {
        return position < numRecords;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public void reset() {
        position = 0;
        if (order != null) {
            RandomUtils.shuffleInPlace(order, rng);
        }
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public Record nextRecord() {
        return next(true);
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<RecordListener> getListeners() {
        return listeners;
    }

    @Override
    public void setListeners(RecordListener... listeners) {
        this.listeners = Arrays.asList(listeners);
    }

    @Override
    public void setListeners(Collection<RecordListener> listeners) {
        this.listeners = new ArrayList<>(listeners);
    }

    @Override
    public void close() throws IOException {
        if (mapFileReader != null) {
            mapFileReader.close();
        }
    }


    private Record next(boolean withMetadata) {
        if (!hasNext()) {
            throw new NoSuchElementException();
        }

        RecordWritable rec;
        long currIdx;
        if (order != null) {
            currIdx = order[(int) position++];
        } else {
            currIdx = position++;
        }

        try {
            rec = mapFileReader.getRecord(currIdx);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        RecordMetaData meta;
        if (withMetadata) {
            meta = new RecordMetaDataIndex(currIdx, uri, MapFileRecordReader.class);
        } else {
            meta = null;
        }

        if(listeners != null && !listeners.isEmpty()){
            for(RecordListener l : listeners){
                l.recordRead(this, rec);
            }
        }

        return new org.datavec.api.records.impl.Record(rec.getRecord(), meta);
    }
}
