package org.datavec.arrow.recordreader;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorLoader;
import org.apache.arrow.vector.VectorUnloader;
import org.apache.arrow.vector.ipc.ArrowFileReader;
import org.apache.arrow.vector.ipc.SeekableReadChannel;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;
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
        if(arrowRecordBatch == null || currIdx >= arrowRecordBatch.getLength()) {
            if(arrowRecordBatch != null) {
                arrowRecordBatch.close();
            }

            String url = pathsIter.next();
            try {
                currIdx = 0;
                InputStream inputStream = split.openInputStreamFor(url);
                try (SeekableReadChannel channel = new SeekableReadChannel(new InputStreamSeekableReadableByteChannel(inputStream,Integer.MAX_VALUE));
                     ArrowFileReader reader = new ArrowFileReader(channel, allocator)) {
                    reader.loadNextBatch();
                    if(this.schema == null) {
                        this.schema = toDatavecSchema(reader.getVectorSchemaRoot().getSchema());
                        fieldVectors = new ArrayList<>(this.schema.numColumns());
                        for(int i = 0; i < schema.numColumns(); i++) {
                            fieldVectors.add(reader.getVectorSchemaRoot().getVector(schema.getName(i)));
                        }
                    }

                    //load the batch
                    VectorUnloader unloader = new VectorUnloader(reader.getVectorSchemaRoot());
                    VectorLoader vectorLoader = new VectorLoader(reader.getVectorSchemaRoot());
                    ArrowRecordBatch recordBatch = unloader.getRecordBatch();

                    vectorLoader.load(recordBatch);
                    this.arrowRecordBatch = recordBatch;


                } catch (IOException e1) {
                    e1.printStackTrace();
                }

            } catch (Exception e) {
                e.printStackTrace();
            }

        }

        if(currIdx < arrowRecordBatch.getLength()) {
            if(recordAllocation == null || recordAllocation.isEmpty()) {
                recordAllocation = new ArrayList<>(schema.numColumns());
                for(int i  = 0; i < schema.numColumns(); i++) {
                    recordAllocation.add(ArrowConverter.fromEntry(currIdx,fieldVectors.get(i),schema.getType(i)));
                }
            }
            else {
                for(int i  = 0; i < schema.numColumns(); i++) {
                    recordAllocation.add(ArrowConverter.fromEntry(currIdx,fieldVectors.get(i),schema.getType(i)));

                }
            }

            currIdx++;
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
