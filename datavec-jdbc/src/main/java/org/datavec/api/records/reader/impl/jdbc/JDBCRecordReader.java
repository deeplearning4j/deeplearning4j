package org.datavec.api.records.reader.impl.jdbc;

import com.zaxxer.hikari.util.DriverDataSource;
import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Properties;
import javax.sql.DataSource;
import lombok.Setter;
import org.apache.commons.dbutils.DbUtils;
import org.apache.commons.dbutils.QueryRunner;
import org.apache.commons.dbutils.handlers.ArrayHandler;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataJdbc;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.jdbc.JdbcWritableConverter;
import org.datavec.api.util.jdbc.ResettableResultSetIterator;
import org.datavec.api.writable.Writable;

/**
 * Iterate on rows from a JDBC datasource and return corresponding records
 *
 * @author Adrien Plagnol
 */
public class JDBCRecordReader extends BaseRecordReader {

    private final String query;
    private Connection conn;
    private Statement statement;
    private ResettableResultSetIterator iter;
    private ResultSetMetaData meta;
    private Configuration configuration;
    @Setter
    private boolean trimStrings = false;
    @Setter
    private DataSource dataSource;
    private final String metadataQuery;
    private final int[] metadataIndices;

    public final static String TRIM_STRINGS = NAME_SPACE + ".trimStrings";
    public final static String JDBC_URL = NAME_SPACE + ".jdbcUrl";
    public final static String JDBC_DRIVER_CLASS_NAME = NAME_SPACE + ".jdbcDriverClassName";
    public final static String JDBC_USERNAME = NAME_SPACE + "jdbcUsername";
    public final static String JDBC_PASSWORD = NAME_SPACE + "jdbcPassword";

    public JDBCRecordReader(String query) {
        this.query = query;
        this.metadataQuery = null;
        this.metadataIndices = null;
    }

    public JDBCRecordReader(String query, DataSource dataSource) {
        this.query = query;
        this.dataSource = dataSource;
        this.metadataQuery = null;
        this.metadataIndices = null;
    }

    public JDBCRecordReader(String query, DataSource dataSource, String metadataQuery, int[] metadataIndices) {
        this.query = query;
        this.dataSource = dataSource;
        this.metadataQuery = metadataQuery;
        this.metadataIndices = metadataIndices;
    }

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        if (dataSource == null) {
            throw new IllegalStateException("Cannot initialize : no datasource");
        }
        initializeJdbc();
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        this.setConf(conf);
        this.trimStrings = conf.getBoolean(TRIM_STRINGS, trimStrings);

        String jdbcUrl = conf.get(JDBC_URL);
        String driverClassName = conf.get(JDBC_DRIVER_CLASS_NAME);
        // url and driver must be both unset or both present
        if (jdbcUrl == null ^ driverClassName == null) {
            throw new IllegalArgumentException(
                "Both jdbc url and driver class name must be provided in order to configure JDBCRecordReader's datasource");
        }
        // Both set, initialiaze the datasource
        else if (jdbcUrl != null) {
            // FIXME : find a way to include wildcard properties as third argument bellow
            this.dataSource = new DriverDataSource(jdbcUrl, driverClassName, new Properties(), conf.get(JDBC_USERNAME),
                conf.get(JDBC_PASSWORD));
            this.initializeJdbc();
        }
    }

    private void initializeJdbc() {
        try {
            this.conn = dataSource.getConnection();
            this.statement = conn.createStatement(ResultSet.TYPE_SCROLL_INSENSITIVE, ResultSet.CONCUR_READ_ONLY);
            this.statement.closeOnCompletion();
            ResultSet rs = statement.executeQuery(this.query);
            this.meta = rs.getMetaData();
            this.iter = new ResettableResultSetIterator(rs);
        } catch (SQLException e) {
            closeJdbc();
            throw new RuntimeException("Could not connect to the database", e);
        }
    }

    @Override
    public List<Writable> next() {
        if (!iter.hasNext()) {
            throw new NoSuchElementException("No next element found!");
        }

        Object[] next = iter.next();
        invokeListeners(next);
        return toWritable(next);
    }

    private List<Writable> toWritable(Object[] item) {
        List<Writable> ret = new ArrayList<>();
        invokeListeners(item);
        for (int i = 0; i < item.length; i++) {
            try {
                Object columnValue = item[i];
                if (trimStrings && columnValue instanceof String) {
                    columnValue = ((String) columnValue).trim();
                }
                // Note, getColumnType first argument is column number starting from 1
                Writable writable = JdbcWritableConverter.convert(columnValue, meta.getColumnType(i + 1));
                ret.add(writable);
            } catch (SQLException e) {
                closeJdbc();
                throw new RuntimeException("Error reading database metadata");
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
        throw new UnsupportedOperationException("JDBCRecordReader does not support getLabels yet");
    }

    @Override
    public void reset() {
        iter.reset();
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException("JDBCRecordReader does not support reading from a DataInputStream");
    }

    @Override
    public Record nextRecord() {
        if (!iter.hasNext()) {
            throw new NoSuchElementException("No next element found!");
        }

        Object[] next = iter.next();
        invokeListeners(next);

        URI location;
        try {
            location = new URI(conn.getMetaData().getURL());
        } catch (SQLException|URISyntaxException e) {
            throw new IllegalStateException("Could not get sql connection metadata", e);
        }

        List<Object> params = new ArrayList<>();
        if (metadataIndices != null) {
            for (int index : metadataIndices) {
                params.add(next[index]);
            }
        }
        RecordMetaDataJdbc rmd = new RecordMetaDataJdbc(location, this.metadataQuery, params, getClass());

        return new org.datavec.api.records.impl.Record(toWritable(next), rmd);
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return loadFromMetaData(Collections.singletonList(recordMetaData)).get(0);
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        List<Record> ret = new ArrayList<>();

        for (RecordMetaData rmd : recordMetaDatas) {
            if (!(rmd instanceof RecordMetaDataJdbc)) {
                throw new IllegalArgumentException(
                    "Invalid metadata; expected RecordMetaDataJdbc instance; got: " + rmd);
            }
            QueryRunner runner = new QueryRunner();
            String request = ((RecordMetaDataJdbc) rmd).getRequest();

            try {
                Object[] item = runner.query(this.conn, request, new ArrayHandler(), ((RecordMetaDataJdbc) rmd).getParams().toArray());
                ret.add(new org.datavec.api.records.impl.Record(toWritable(item), rmd));
            } catch (SQLException e) {
                throw new IllegalArgumentException("Could not execute statement \"" + request + "\"", e);
            }
        }
        return ret;
    }

    @Override
    public void close() throws IOException {
        closeJdbc();
    }

    private void closeJdbc() {
        DbUtils.closeQuietly(statement);
        DbUtils.closeQuietly(conn);
    }

    @Override
    public void setConf(Configuration conf) {
        this.configuration = conf;
    }

    @Override
    public Configuration getConf() {
        return this.configuration;
    }
}
