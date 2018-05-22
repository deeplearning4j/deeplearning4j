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
    private int resultSetType = ResultSet.TYPE_SCROLL_INSENSITIVE;
    @Setter
    private DataSource dataSource;
    private final String metadataQuery;
    private final int[] metadataIndices;

    public final static String TRIM_STRINGS = NAME_SPACE + ".trimStrings";
    public final static String JDBC_URL = NAME_SPACE + ".jdbcUrl";
    public final static String JDBC_DRIVER_CLASS_NAME = NAME_SPACE + ".jdbcDriverClassName";
    public final static String JDBC_USERNAME = NAME_SPACE + ".jdbcUsername";
    public final static String JDBC_PASSWORD = NAME_SPACE + ".jdbcPassword";
    public final static String JDBC_RESULTSET_TYPE = NAME_SPACE + ".resultSetType";

    /**
     * Build a new JDBCRecordReader with a given query. After constructing the reader in this way, the initialize method
     * must be called and provided with configuration values for the datasource initialization.
     *
     * @param query Query to execute and on which the reader will iterate.
     */
    public JDBCRecordReader(String query) {
        this.query = query;
        this.metadataQuery = null;
        this.metadataIndices = null;
    }

    /**
     * Build a new JDBCRecordReader with a given query. If initialize is called with configuration values set for
     * datasource initialization, the datasource provided to this constructor will be overriden.
     *
     * @param query Query to execute and on which the reader will iterate.
     * @param dataSource Initialized DataSource to use for iteration
     */
    public JDBCRecordReader(String query, DataSource dataSource) {
        this.query = query;
        this.dataSource = dataSource;
        this.metadataQuery = null;
        this.metadataIndices = null;
    }

    /**
     * Same as JDBCRecordReader(String query, DataSource dataSource) but also provides a query and column indices to use
     * for saving metadata (see {@link #loadFromMetaData(RecordMetaData)})
     *
     * @param query Query to execute and on which the reader will iterate.
     * @param dataSource Initialized DataSource to use for iteration.
     * @param metadataQuery Query to execute when recovering a single record from metadata
     * @param metadataIndices Column indices of which values will be saved in each record's metadata
     */
    public JDBCRecordReader(String query, DataSource dataSource, String metadataQuery, int[] metadataIndices) {
        this.query = query;
        this.dataSource = dataSource;
        this.metadataQuery = metadataQuery;
        this.metadataIndices = metadataIndices;
    }

    /**
     * Initialize all required jdbc elements and make the reader ready for iteration.
     *
     * @param split not handled yet, will be discarded
     */
    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        if (dataSource == null) {
            throw new IllegalStateException("Cannot initialize : no datasource");
        }
        initializeJdbc();
    }

    /**
     * Initialize all required jdbc elements and make the reader ready for iteration.
     *
     * Possible configuration keys :
     * <ol>
     *     <li>JDBCRecordReader.TRIM_STRINGS : Whether or not read strings should be trimmed before being returned. False by default</li>
     *     <li>JDBCRecordReader.JDBC_URL : Jdbc url to use for datastource configuration (see JDBCRecordReaderTest for examples)</li>
     *     <li>JDBCRecordReader.JDBC_DRIVER_CLASS_NAME : Driver class to use for datasource configuration</li>
     *     <li>JDBCRecordReader.JDBC_USERNAME && JDBC_PASSWORD : Username and password to use for datasource configuration</li>
     *     <li>JDBCRecordReader.JDBC_RESULTSET_TYPE : ResultSet type to use (int value defined in jdbc doc)</li>
     * </ol>
     *
     * Url and driver class name are not mandatory. If one of them is specified, the other must be specified as well. If
     * they are set and there already is a DataSource set in the reader, it will be discarded and replaced with the
     * newly created one.
     *
     * @param conf a configuration for initialization
     * @param split not handled yet, will be discarded
     */
    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        this.setConf(conf);
        this.setTrimStrings(conf.getBoolean(TRIM_STRINGS, trimStrings));
        this.setResultSetType(conf.getInt(JDBC_RESULTSET_TYPE, resultSetType));

        String jdbcUrl = conf.get(JDBC_URL);
        String driverClassName = conf.get(JDBC_DRIVER_CLASS_NAME);
        // url and driver must be both unset or both present
        if (jdbcUrl == null ^ driverClassName == null) {
            throw new IllegalArgumentException(
                "Both jdbc url and driver class name must be provided in order to configure JDBCRecordReader's datasource");
        }
        // Both set, initialiaze the datasource
        else if (jdbcUrl != null) {
            // FIXME : find a way to read wildcard properties from conf in order to fill the third argument bellow
            this.dataSource = new DriverDataSource(jdbcUrl, driverClassName, new Properties(), conf.get(JDBC_USERNAME),
                conf.get(JDBC_PASSWORD));
        }
        this.initializeJdbc();
    }

    private void initializeJdbc() {
        try {
            this.conn = dataSource.getConnection();
            this.statement = conn.createStatement(this.resultSetType, ResultSet.CONCUR_READ_ONLY);
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

    /**
     * Depending on the jdbc driver implementation, this will probably fail if the resultset was created with ResultSet.TYPE_FORWARD_ONLY
     */
    @Override
    public void reset() {
        iter.reset();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException("JDBCRecordReader does not support reading from a DataInputStream");
    }

    /**
     * Get next record with metadata. See {@link #loadFromMetaData(RecordMetaData)} for details on metadata structure.
     */
    @Override
    public Record nextRecord() {
        Object[] next = iter.next();
        invokeListeners(next);

        URI location;
        try {
            location = new URI(conn.getMetaData().getURL());
        } catch (SQLException | URISyntaxException e) {
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

    /**
     * Record metadata for this reader consist in two elements :<br />
     *
     * - a parametrized query used to retrieve one item<br />
     *
     * - a set a values to use to prepare the statement<br /><br />
     *
     * The parametrized query is passed at construction time and it should fit the main record's reader query. For
     * instance, one could have to following reader query : "SELECT * FROM Items", and a corresponding metadata query
     * could be "SELECT * FROM Items WHERE id = ?". For each record, the columns indicated in {@link #metadataIndices}
     * will be stored. For instance, one could set metadataIndices = {0} so the value of the first column of each record
     * is stored in the metadata.
     *
     * @param recordMetaData Metadata for the record that we want to load from
     */
    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return loadFromMetaData(Collections.singletonList(recordMetaData)).get(0);
    }

    /**
     * @see #loadFromMetaData(RecordMetaData)
     */
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
                Object[] item = runner
                    .query(this.conn, request, new ArrayHandler(), ((RecordMetaDataJdbc) rmd).getParams().toArray());
                ret.add(new org.datavec.api.records.impl.Record(toWritable(item), rmd));
            } catch (SQLException e) {
                throw new IllegalArgumentException("Could not execute statement \"" + request + "\"", e);
            }
        }
        return ret;
    }

    /**
     * Expected to be called by the user. JDBC connections will not be closed automatically.
     */
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
