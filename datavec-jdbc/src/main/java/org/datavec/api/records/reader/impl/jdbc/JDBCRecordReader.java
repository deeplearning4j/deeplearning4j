package org.datavec.api.records.reader.impl.jdbc;

import com.zaxxer.hikari.util.DriverDataSource;
import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import javax.sql.DataSource;
import lombok.Setter;
import org.apache.commons.dbutils.DbUtils;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
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
    private DataSource dataSource;
    private Connection conn;
    private Statement statement;
    private ResettableResultSetIterator iter;
    private ResultSetMetaData meta;
    @Setter
    private boolean trimStrings = false;

    public final static String TRIM_STRINGS = NAME_SPACE + ".trimStrings";
    public final static String JDBC_URL = NAME_SPACE + ".jdbcUrl";
    public final static String JDBC_DRIVER_CLASS_NAME = NAME_SPACE + ".jdbcDriverClassName";
    public final static String JDBC_USERNAME = NAME_SPACE + "jdbcUsername" ;
    public final static String JDBC_PASSWORD = NAME_SPACE + "jdbcPassword";

    public JDBCRecordReader(String query) {
        this.query = query;
    }

    public JDBCRecordReader(String query, DataSource dataSource) {
        this.dataSource = dataSource;
        this.query = query;
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
        this.trimStrings = conf.getBoolean(TRIM_STRINGS, trimStrings);
        String jdbcUrl = conf.get(JDBC_URL);
        String driverClassName = conf.get(JDBC_DRIVER_CLASS_NAME);
        // url and driver must be both unset or both present
        if (jdbcUrl == null ^ driverClassName == null) {
            throw new IllegalArgumentException("Both jdbc url and driver class name must be provided in order to configure JDBCRecordReader's datasource");
        }
        // Both set, initialiaze the datasource
        else if (jdbcUrl != null) {
            // FIXME : find a way to include wildcard properties as third argument bellow
            this.dataSource = new DriverDataSource(jdbcUrl, driverClassName, new Properties(), conf.get("username"), conf.get("password"));
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
        List<Writable> ret = new ArrayList<>();
        if (iter.hasNext()) {
            Object[] next = iter.next();
            invokeListeners(next);
            for (int i = 0; i < next.length; i++) {
                try {
                    Object columnValue = next[i];
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
        iter.reset();
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
        closeJdbc();
    }

    private void closeJdbc() {
        DbUtils.closeQuietly(statement);
        DbUtils.closeQuietly(conn);
    }

    @Override
    public void setConf(Configuration conf) {

    }

    @Override
    public Configuration getConf() {
        return null;
    }
}
