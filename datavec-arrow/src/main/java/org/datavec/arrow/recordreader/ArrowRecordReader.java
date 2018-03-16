package org.datavec.arrow.recordreader;

import lombok.val;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorLoader;
import org.apache.arrow.vector.VectorUnloader;
import org.apache.arrow.vector.ipc.ArrowFileReader;
import org.apache.arrow.vector.ipc.SeekableReadChannel;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;
import org.apache.arrow.vector.util.ByteArrayReadableSeekableByteChannel;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.IOUtils;
import org.datavec.api.records.Record;
import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.arrow.ArrowConverter;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import static org.datavec.arrow.ArrowConverter.readFromBytes;
import static org.datavec.arrow.ArrowConverter.toDatavecSchema;

/**
 * Implements a record reader using arrow.
 *
 */
public class ArrowRecordReader implements RecordReader {

    private InputSplit split;
    private Configuration configuration;
    private BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    private Iterator<String> pathsIter;
    private ArrowRecordBatch arrowRecordBatch;
    private int currIdx;
    private List<FieldVector> fieldVectors;
    private Schema schema;
    private List<Writable> recordAllocation = new ArrayList<>();
    private List<List<Writable>> currentBatch;
    @Override
    public void initialize(InputSplit split) {
        this.split = split;
        this.pathsIter = split.locationsPathIterator();
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) {
        this.split = split;
        this.pathsIter = split.locationsPathIterator();

    }

    @Override
    public boolean batchesSupported() {
        return true;
    }

    @Override
    public List<Writable> next(int num) {
        return next();
    }

    @Override
    public List<Writable> next() {
        if (currentBatch == null || currIdx >= currentBatch.size()) {
            String url = pathsIter.next();
            try (InputStream inputStream = split.openInputStreamFor(url)) {
                currIdx = 0;
                byte[] arr = org.apache.commons.io.IOUtils.toByteArray(inputStream);
                val read = readFromBytes(arr);
                if(this.schema == null) {
                    this.schema = read.getFirst();
                }

                this.currentBatch = read.getRight();
                this.recordAllocation = currentBatch.get(0);
                currIdx++;
            }catch(Exception e) {
                e.printStackTrace();
            }


        }
        else {
            recordAllocation = currentBatch.get(currIdx++);
        }

        return recordAllocation;

    }

    @Override
    public boolean hasNext() {
        return pathsIter.hasNext() || currIdx < arrowRecordBatch.getLength();
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public void reset() {
        if(split != null) {
            split.reset();
        }
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) {
        return null;
    }

    @Override
    public Record nextRecord() {
        return null;
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) {
        return null;
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) {
        return null;
    }

    @Override
    public List<RecordListener> getListeners() {
        return null;
    }

    @Override
    public void setListeners(RecordListener... listeners) {

    }

    @Override
    public void setListeners(Collection<RecordListener> listeners) {

    }

    @Override
    public void close() {

    }

    @Override
    public void setConf(Configuration conf) {
        this.configuration = conf;
    }

    @Override
    public Configuration getConf() {
        return configuration;
    }
}
