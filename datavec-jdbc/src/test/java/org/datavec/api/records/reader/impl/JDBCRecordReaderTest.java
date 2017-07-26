package org.datavec.api.records.reader.impl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.net.URI;
import java.sql.Connection;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.commons.dbutils.DbUtils;
import org.apache.derby.jdbc.EmbeddedDataSource;
import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.records.reader.impl.jdbc.JDBCRecordReader;
import org.datavec.api.split.CollectionInputSplit;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class JDBCRecordReaderTest {

    Connection conn;
    EmbeddedDataSource dataSource;

    @Before
    public void setUp() throws Exception {
        dataSource = new EmbeddedDataSource();
        dataSource.setDatabaseName("datavecTests");
        dataSource.setCreateDatabase("create");
        conn = dataSource.getConnection();

        TestDb.dropTables(conn);
        TestDb.buildCoffeeTable(conn);
    }

    @After
    public void tearDown() throws Exception {
        DbUtils.closeQuietly(conn);
    }

    @Test
    public void testSimpleIter() throws Exception {
        JDBCRecordReader reader = getInitializedReader("SELECT * FROM Coffee");
        List<List<Writable>> records = new ArrayList<>();
        while (reader.hasNext()) {
            List<Writable> values = reader.next();
            records.add(values);
        }
        reader.close();
        assertFalse(records.isEmpty());

        List<Writable> first = records.get(0);
        assertEquals(new Text("Bolivian Dark"), first.get(0));
        assertEquals(new Text("14-001"), first.get(1));
        assertEquals(new DoubleWritable(8.95), first.get(2));
    }

    @Test
    public void testSimpleWithListener() throws Exception {
        JDBCRecordReader reader = getInitializedReader("SELECT * FROM Coffee");
        RecordListener recordListener = new LogRecordListener();
        reader.setListeners(recordListener);
        reader.next();
        reader.close();
        assertTrue(recordListener.invoked());
    }

    private JDBCRecordReader getInitializedReader(String query) throws Exception {
        JDBCRecordReader reader = new JDBCRecordReader(dataSource, query);
        reader.setTrimStrings(true);
        // FIXME should implement a new input split ?
        reader.initialize(new CollectionInputSplit(Collections.<URI>emptyList()));
        return reader;
    }
}