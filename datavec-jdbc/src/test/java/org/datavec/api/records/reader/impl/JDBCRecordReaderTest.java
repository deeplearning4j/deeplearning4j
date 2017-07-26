package org.datavec.api.records.reader.impl;

import org.apache.derby.jdbc.BasicEmbeddedDataSource40;
import org.apache.derby.jdbc.EmbeddedDataSource;
import org.datavec.api.records.reader.impl.jdbc.JDBCRecordReader;
import org.datavec.api.writable.Writable;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import javax.sql.DataSource;

import java.sql.Connection;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.*;

public class JDBCRecordReaderTest {

    Connection conn;
    EmbeddedDataSource dataSource;

    @Before
    public void setUp() throws Exception {
        dataSource = new EmbeddedDataSource();
        dataSource.setDatabaseName("datavecTests");
        dataSource.setCreateDatabase("create");
        conn = dataSource.getConnection();

        TestDb.buildCoffeeTable(conn);
    }

    @After
    public void tearDown() throws Exception {
        TestDb.dropTables(conn);
    }

    @Test
    public void testSimpleIter() throws Exception {
        JDBCRecordReader recordReader = new JDBCRecordReader(dataSource, "SELECT * FROM Coffee;");
        List<List<Writable>> records = Collections.emptyList();
        while(recordReader.hasNext()) {
            List<Writable> values = recordReader.next();
            records.add(values);
        }
        assertFalse(records.isEmpty());

        List<Writable> first = records.get(0);
        assertEquals("Bolivian Dark", first.get(0));
        assertEquals("14-001", first.get(1));
        assertEquals(8.95, first.get(2));
    }


}