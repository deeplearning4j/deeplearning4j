package org.deeplearning4j.spark.datavec.iterator;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.Serializable;
import java.net.URI;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

@AllArgsConstructor
@Data
public class SparkSourceDummyReader implements RecordReader, Serializable {

    private int readerIdx;


    @Override
    public void initialize(InputSplit inputSplit) throws IOException, InterruptedException {
        /* No op */
    }

    @Override
    public void initialize(Configuration configuration, InputSplit inputSplit) throws IOException, InterruptedException {
        /* No op */
    }

    @Override
    public boolean batchesSupported() {
        return false;
    }

    @Override
    public List<Writable> next(int i) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<Writable> next() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean hasNext() {
        return false;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public void reset() { /* No op */ }

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
    public List<Record> loadFromMetaData(List<RecordMetaData> list) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<RecordListener> getListeners() { return Collections.emptyList(); }

    @Override
    public void setListeners(RecordListener... recordListeners) { /* No op */ }

    @Override
    public void setListeners(Collection<RecordListener> collection) { }

    @Override
    public void close() throws IOException { /* No op */}

    @Override
    public void setConf(Configuration configuration) { /* No op */ }

    @Override
    public Configuration getConf() { throw new UnsupportedOperationException(); }
}
