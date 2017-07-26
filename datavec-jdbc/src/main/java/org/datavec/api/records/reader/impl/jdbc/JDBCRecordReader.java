package org.datavec.api.records.reader.impl.jdbc;

import lombok.Setter;
import org.apache.commons.dbutils.ResultSetIterator;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.jdbc.JdbcWritableConverter;
import org.datavec.api.writable.Writable;

import javax.sql.DataSource;
import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.sql.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Iterate on rows from a JDBC datasource and return corresponding records
 *
 * @author Adrien Plagnol
 */
public class JDBCRecordReader extends BaseRecordReader {

    private final DataSource dataSource;
    private final String query;
    private ResultSetIterator iter;
    private ResultSetMetaData meta;
    @Setter
    private boolean trimStrings = false;

    public JDBCRecordReader(DataSource dataSource, String query) {
        this.dataSource = dataSource;
        this.query = query;
    }

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        try {
            Connection conn = dataSource.getConnection();
            Statement st = conn.createStatement(ResultSet.TYPE_SCROLL_INSENSITIVE, ResultSet.CONCUR_READ_ONLY);
            ResultSet rs = st.executeQuery(this.query);
            this.meta = rs.getMetaData();
            this.iter = new ResultSetIterator(rs);
        } catch (SQLException e) {
            throw new RuntimeException("Could not connect to the database", e);
        }
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {

    }

    @Override
    public List<Writable> next() {
        List<Writable> ret = new ArrayList<>();
        if (iter.hasNext()) {
            Object[] next = iter.next();
            for (int i = 0; i < next.length; i++) {
                try {
                    Object columnValue = next[i];
                    if (trimStrings && columnValue instanceof String) {
                        columnValue = ((String) columnValue).trim();
                    }
                    // Note, getColumnType first argument is column number starting from 1
                    Writable writable = JdbcWritableConverter.convert(columnValue, meta.getColumnType(i+1));
                    ret.add(writable);
                } catch (SQLException e) {
                    throw new RuntimeException("Error reading database metadata");
                }
            }
        }

        return ret;
    }

    @Override
    public boolean hasNext() {
        return iter.hasNext();
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
