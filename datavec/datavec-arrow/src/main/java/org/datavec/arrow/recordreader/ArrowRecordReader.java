package org.datavec.arrow.recordreader;

import lombok.Getter;
import lombok.val;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataIndex;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;

import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.util.*;

import static org.datavec.arrow.ArrowConverter.readFromBytes;

/**
 * Implements a record reader using arrow.
 * The {@link ArrowRecordReader} minimizes memory footprint by
 * using an {@link ArrowWritableRecordBatch} as the current in memory
 * batch during iteration rather than the normal of objects
 * you would find with the traditional record readers with {@link List<List<Writable>>}
 *
 *
 *
 * @author Adam Gibson
 *
 */
public class ArrowRecordReader implements RecordReader {

    private InputSplit split;
    private Configuration configuration;
    private Iterator<String> pathsIter;
    private int currIdx;
    private String currentPath;
    private Schema schema;
    private List<Writable> recordAllocation = new ArrayList<>();
    @Getter
    private ArrowWritableRecordBatch currentBatch;
    private List<RecordListener> recordListeners;

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
    public List<List<Writable>> next(int num) {
        if (currentBatch == null || currIdx >= currentBatch.size()) {
            loadNextBatch();
        }

        if(num == currentBatch.getArrowRecordBatch().getLength()) {
            currIdx += num;
            return currentBatch;
        }
        else {
            List<List<Writable>> ret = new ArrayList<>(num);
            int numBatches = 0;
            while(hasNext() && numBatches < num) {
                ret.add(next());
            }

            return ret;
        }


    }

    @Override
    public List<Writable> next() {
        if (currentBatch == null || currIdx >= currentBatch.size()) {
            loadNextBatch();
        }
        else {
            recordAllocation = currentBatch.get(currIdx++);
        }

        return recordAllocation;

    }

    private void loadNextBatch() {
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
            this.currentPath = url;
        }catch(Exception e) {
            e.printStackTrace();
        }

    }


    @Override
    public boolean hasNext() {
        return pathsIter.hasNext() || currIdx < this.currentBatch.size();
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException();
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
        throw new UnsupportedOperationException();
    }

    @Override
    public Record nextRecord() {
        next();
        ArrowRecord ret =  new ArrowRecord(currentBatch,currIdx - 1,URI.create(currentPath));
        return ret;
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) {
        if(!(recordMetaData instanceof RecordMetaDataIndex)) {
            throw new IllegalArgumentException("Unable to load from meta data. No index specified for record");
        }

        RecordMetaDataIndex index = (RecordMetaDataIndex) recordMetaData;
        InputSplit fileSplit = new FileSplit(new File(index.getURI()));
        initialize(fileSplit);
        this.currIdx = (int) index.getIndex();
        return nextRecord();
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) {
        Map<String,List<RecordMetaData>> metaDataByUri = new HashMap<>();
        //gather all unique locations for the metadata
        //this will prevent initialization multiple times of the record
        for(RecordMetaData recordMetaData : recordMetaDatas) {
            if(!(recordMetaData instanceof RecordMetaDataIndex)) {
                throw new IllegalArgumentException("Unable to load from meta data. No index specified for record");
            }

            List<RecordMetaData> recordMetaData1 = metaDataByUri.get(recordMetaData.getURI().toString());
            if(recordMetaData1 == null) {
                recordMetaData1 = new ArrayList<>();
                metaDataByUri.put(recordMetaData.getURI().toString(),recordMetaData1);
            }

            recordMetaData1.add(recordMetaData);

        }

        List<Record> ret = new ArrayList<>();
        for(String uri : metaDataByUri.keySet()) {
            List<RecordMetaData> metaData = metaDataByUri.get(uri);
            InputSplit fileSplit = new FileSplit(new File(URI.create(uri)));
            initialize(fileSplit);
            for(RecordMetaData index : metaData) {
                RecordMetaDataIndex index2 = (RecordMetaDataIndex) index;
                this.currIdx = (int) index2.getIndex();
                ret.add(nextRecord());
            }

        }

        return ret;
    }

    @Override
    public List<RecordListener> getListeners() {
        return recordListeners;
    }

    @Override
    public void setListeners(RecordListener... listeners) {
        this.recordListeners = new ArrayList<>(Arrays.asList(listeners));
    }

    @Override
    public void setListeners(Collection<RecordListener> listeners) {
        this.recordListeners = new ArrayList<>(listeners);
    }

    @Override
    public void close() {
        if(currentBatch != null) {
            try {
                currentBatch.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
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
