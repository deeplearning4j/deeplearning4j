package org.datavec.api.records.reader.impl.jdbc;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;

import javax.sql.DataSource;
import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.sql.ResultSet;
import java.util.List;

public class JDBCRecordReader extends BaseRecordReader {

    private DataSource dataSource;
    private ResultSet resultSet;
    private String query;

    public JDBCRecordReader(DataSource dataSource, String query) {
        this.dataSource = dataSource;
        this.query = query;
    }

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {

    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {

    }

    @Override
    public List<Writable> next() {
        return null;
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
    public void reset() {

    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        return null;
    }

    @Override
    public Record nextRecord() {
        return null;
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return null;
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        return null;
    }

    @Override
    public void close() throws IOException {

    }

    @Override
    public void setConf(Configuration conf) {

    }

    @Override
    public Configuration getConf() {
        return null;
    }
}
