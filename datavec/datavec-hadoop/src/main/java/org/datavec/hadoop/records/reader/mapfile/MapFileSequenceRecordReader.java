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

package org.datavec.hadoop.records.reader.mapfile;

import lombok.NonNull;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataIndex;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;
import org.datavec.hadoop.records.reader.mapfile.index.LongIndexToKey;
import org.datavec.hadoop.records.reader.mapfile.record.SequenceRecordWritable;
import org.nd4j.linalg.util.MathUtils;

import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.*;

/**
 * A {@link SequenceRecordReader} implementation for reading from a Hadoop {@link org.apache.hadoop.io.MapFile}<br>
 * <p>
 * A typical use case is with {@link org.datavec.api.transform.TransformProcess} executed on Spark (perhaps Spark
 * local), followed by non-distributed training on a single machine. For example:
 * <pre>
 *  {@code
 *  JavaRDD<List<List<Writable>>> myRDD = ...;
 *  String mapFilePath = ...;
 *  SparkStorageUtils.saveMapFileSequences( mapFilePath, myRDD );
 *
 *  SequenceRecordReader rr = new MapFileSequenceRecordReader();
 *  rr.initialize( new FileSplit( new File( mapFilePath ) ) );
 *  //Pass to DataSetIterator or similar
 *  }
 * </pre>
 *
 * Alternatively, use {@link org.datavec.hadoop.records.writer.mapfile.MapFileSequenceRecordWriter}.<br>
 * Note that this sequence record reader supports optional randomisation of order.
 *
 * @author Alex Black
 */
public class MapFileSequenceRecordReader implements SequenceRecordReader {
    private static final Class<? extends org.apache.hadoop.io.Writable> recordClass = SequenceRecordWritable.class;

    private final IndexToKey indexToKey;
    private MapFileReader<SequenceRecordWritable> mapFileReader;
    private URI baseDirUri;
    private List<RecordListener> listeners;

    private long numSequences;
    private long position;
    private Random rng;
    private int[] order;

    /**
     * Create a MapFileSequenceRecordReader with no randomisation, and assuming MapFile keys are {@link org.apache.hadoop.io.LongWritable}
     * values
     */
    public MapFileSequenceRecordReader() {
        this(new LongIndexToKey(), null);
    }

    /**
     * Create a MapFileSequenceRecordReader with optional randomisation, and assuming MapFile keys are
     * {@link org.apache.hadoop.io.LongWritable} values
     *
     * @param rng If non-null, will be used to randomize the order of examples
     *
     */
    public MapFileSequenceRecordReader(Random rng) {
        this(new LongIndexToKey(), rng);
    }

    /**
     * Create a MapFileSequenceRecordReader with optional randomisation, with a custom {@link IndexToKey} instance to
     * handle MapFile keys
     *
     * @param indexToKey Handles conversion between long indices and key values (see for example {@link LongIndexToKey}
     * @param rng If non-null, will be used to randomize the order of examples
     *
     */
    public MapFileSequenceRecordReader(IndexToKey indexToKey, Random rng) {
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

        //First: work out whether we have a single MapFile or multiple parts
        int dataCount = 0;
        int indexCount = 0;
        List<URI> dataUris = new ArrayList<>();
        for (URI u : uris) {
            String p = u.getPath();
            if (p.endsWith("data")) {
                dataCount++;
                dataUris.add(u);
            } else if (p.endsWith("index")) {
                indexCount++;
            }
        }

        //Check URIs are correct: we expect one or more /data and /index files...
        if (dataCount == 0 || indexCount == 0) {
            throw new IllegalStateException("Cannot initialize MapFileSequenceRecordReader: could not find data and "
                            + "index files in input split");
        }
        if (dataCount != indexCount) {
            throw new IllegalStateException("Invalid input: found " + dataCount + " data files but " + indexCount
                            + " index files. Expect equal number of both for map files");
        }

        List<String> mapFilePartRootDirectories = new ArrayList<>(dataUris.size());
        for (URI u : dataUris) {
            File partRootDir = new File(u).getParentFile();
            mapFilePartRootDirectories.add(partRootDir.getAbsolutePath());
        }

        //Sort the paths so we iterate over multi-part MapFiles like part-r-00000, part-r-00001, etc when not randomized
        Collections.sort(mapFilePartRootDirectories);


        if (dataUris.size() == 1) {
            //Just parent of /data
            baseDirUri = new File(dataUris.get(0)).getParentFile().toURI();
        } else {
            //Multiple parts -> up 2 levels from data
            //so, /baseDir/part-r-00000/data -> /baseDir
            baseDirUri = new File(dataUris.get(0)).getParentFile().getParentFile().toURI();
        }

        if (mapFileReader != null) {
            mapFileReader.close();
        }

        this.mapFileReader = new MapFileReader<>(mapFilePartRootDirectories, indexToKey, recordClass);
        this.numSequences = mapFileReader.numRecords();

        if (rng != null) {
            order = new int[(int) numSequences];
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            MathUtils.shuffleArray(order, rng);
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
    public List<List<Writable>> sequenceRecord() {
        return nextSequence(false).getSequenceRecord();
    }

    @Override
    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException("MapFileSequenceRecordReader: does not support reading from streams");
    }

    @Override
    public SequenceRecord nextSequence() {
        return nextSequence(true);
    }

    private SequenceRecord nextSequence(boolean withMetadata) {
        if (!hasNext()) {
            throw new NoSuchElementException();
        }

        SequenceRecordWritable seq;
        long currIdx;
        if (order != null) {
            currIdx = order[(int) position++];
        } else {
            currIdx = position++;
        }

        try {
            seq = mapFileReader.getRecord(currIdx);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        RecordMetaData meta;
        if (withMetadata) {
            meta = new RecordMetaDataIndex(currIdx, baseDirUri, MapFileSequenceRecordReader.class);
        } else {
            meta = null;
        }

        if (listeners != null && !listeners.isEmpty()) {
            for (RecordListener l : listeners) {
                l.recordRead(this, seq);
            }
        }

        return new org.datavec.api.records.impl.SequenceRecord(seq.getSequenceRecord(), meta);
    }

    @Override
    public SequenceRecord loadSequenceFromMetaData(@NonNull RecordMetaData recordMetaData) throws IOException {
        long idx = ((RecordMetaDataIndex) recordMetaData).getIndex();
        return new org.datavec.api.records.impl.SequenceRecord(mapFileReader.getRecord(idx).getSequenceRecord(),
                        recordMetaData);
    }

    @Override
    public List<SequenceRecord> loadSequenceFromMetaData(@NonNull List<RecordMetaData> recordMetaDatas)
                    throws IOException {
        List<SequenceRecord> out = new ArrayList<>(recordMetaDatas.size());
        for (RecordMetaData r : recordMetaDatas) {
            out.add(loadSequenceFromMetaData(r));
        }
        return out;
    }

    @Override
    public boolean batchesSupported() {
        return false;
    }

    @Override
    public List<List<Writable>> next(int num) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<Writable> next() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean hasNext() {
        return position < numSequences;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public void reset() {
        position = 0;
        if (order != null) {
            MathUtils.shuffleArray(order, rng);
        }
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public Record nextRecord() {
        throw new UnsupportedOperationException();
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
}
