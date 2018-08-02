/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.api.records.reader.impl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.net.URI;
import java.sql.Connection;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.commons.dbutils.DbUtils;
import org.apache.derby.jdbc.EmbeddedDataSource;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataJdbc;
import org.datavec.api.records.metadata.RecordMetaDataLine;
import org.datavec.api.records.reader.impl.jdbc.JDBCRecordReader;
import org.datavec.api.writable.BooleanWritable;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.FloatWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class JDBCRecordReaderTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    Connection conn;
    EmbeddedDataSource dataSource;

    private final String dbName = "datavecTests";
    private final String driverClassName = "org.apache.derby.jdbc.EmbeddedDriver";

    @Before
    public void setUp() throws Exception {
        File f = testDir.newFolder();
        System.setProperty("derby.system.home", f.getAbsolutePath());

        dataSource = new EmbeddedDataSource();
        dataSource.setDatabaseName(dbName);
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
        try (JDBCRecordReader reader = getInitializedReader("SELECT * FROM Coffee")) {
            List<List<Writable>> records = new ArrayList<>();
            while (reader.hasNext()) {
                List<Writable> values = reader.next();
                records.add(values);
            }

            assertFalse(records.isEmpty());

            List<Writable> first = records.get(0);
            assertEquals(new Text("Bolivian Dark"), first.get(0));
            assertEquals(new Text("14-001"), first.get(1));
            assertEquals(new DoubleWritable(8.95), first.get(2));
        }
    }

    @Test
    public void testSimpleWithListener() throws Exception {
        try (JDBCRecordReader reader = getInitializedReader("SELECT * FROM Coffee")) {
            RecordListener recordListener = new LogRecordListener();
            reader.setListeners(recordListener);
            reader.next();

            assertTrue(recordListener.invoked());
        }
    }

    @Test
    public void testReset() throws Exception {
        try (JDBCRecordReader reader = getInitializedReader("SELECT * FROM Coffee")) {
            List<List<Writable>> records = new ArrayList<>();
            records.add(reader.next());
            reader.reset();
            records.add(reader.next());

            assertEquals(2, records.size());
            assertEquals(new Text("Bolivian Dark"), records.get(0).get(0));
            assertEquals(new Text("Bolivian Dark"), records.get(1).get(0));
        }
    }

    @Test(expected = IllegalStateException.class)
    public void testLackingDataSourceShouldFail() throws Exception {
        try (JDBCRecordReader reader = new JDBCRecordReader("SELECT * FROM Coffee")) {
            reader.initialize(null);
        }
    }

    @Test
    public void testConfigurationDataSourceInitialization() throws Exception {
        try (JDBCRecordReader reader = new JDBCRecordReader("SELECT * FROM Coffee")) {
            Configuration conf = new Configuration();
            conf.set(JDBCRecordReader.JDBC_URL, "jdbc:derby:" + dbName + ";create=true");
            conf.set(JDBCRecordReader.JDBC_DRIVER_CLASS_NAME, driverClassName);
            reader.initialize(conf, null);
            assertTrue(reader.hasNext());
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInitConfigurationMissingParametersShouldFail() throws Exception {
        try (JDBCRecordReader reader = new JDBCRecordReader("SELECT * FROM Coffee")) {
            Configuration conf = new Configuration();
            conf.set(JDBCRecordReader.JDBC_URL, "should fail anyway");
            reader.initialize(conf, null);
        }
    }

    @Test(expected = UnsupportedOperationException.class)
    public void testRecordDataInputStreamShouldFail() throws Exception {
        try (JDBCRecordReader reader = getInitializedReader("SELECT * FROM Coffee")) {
            reader.record(null, null);
        }
    }

    @Test
    public void testLoadFromMetaData() throws Exception {
        try (JDBCRecordReader reader = getInitializedReader("SELECT * FROM Coffee")) {
            RecordMetaDataJdbc rmd = new RecordMetaDataJdbc(new URI(conn.getMetaData().getURL()),
                "SELECT * FROM Coffee WHERE ProdNum = ?", Collections.singletonList("14-001"), reader.getClass());

            Record res = reader.loadFromMetaData(rmd);
            assertNotNull(res);
            assertEquals(new Text("Bolivian Dark"), res.getRecord().get(0));
            assertEquals(new Text("14-001"), res.getRecord().get(1));
            assertEquals(new DoubleWritable(8.95), res.getRecord().get(2));
        }
    }

    @Test
    public void testNextRecord() throws Exception {
        try (JDBCRecordReader reader = getInitializedReader("SELECT * FROM Coffee")) {
            Record r = reader.nextRecord();
            List<Writable> fields = r.getRecord();
            RecordMetaData meta = r.getMetaData();
            assertNotNull(r);
            assertNotNull(fields);
            assertNotNull(meta);
            assertEquals(new Text("Bolivian Dark"), fields.get(0));
            assertEquals(new Text("14-001"), fields.get(1));
            assertEquals(new DoubleWritable(8.95), fields.get(2));
            assertEquals(RecordMetaDataJdbc.class, meta.getClass());
        }
    }

    @Test
    public void testNextRecordAndRecover() throws Exception {
        try (JDBCRecordReader reader = getInitializedReader("SELECT * FROM Coffee")) {
            Record r = reader.nextRecord();
            List<Writable> fields = r.getRecord();
            RecordMetaData meta = r.getMetaData();
            Record recovered = reader.loadFromMetaData(meta);
            List<Writable> fieldsRecovered = recovered.getRecord();
            assertEquals(fields.size(), fieldsRecovered.size());
            for (int i = 0; i < fields.size(); i++) {
                assertEquals(fields.get(i), fieldsRecovered.get(i));
            }
        }
    }

    // Resetting the record reader when initialized as forward only should fail
    @Test(expected = RuntimeException.class)
    public void testResetForwardOnlyShouldFail() throws Exception {
        try (JDBCRecordReader reader = new JDBCRecordReader("SELECT * FROM Coffee", dataSource)) {
            Configuration conf = new Configuration();
            conf.setInt(JDBCRecordReader.JDBC_RESULTSET_TYPE, ResultSet.TYPE_FORWARD_ONLY);
            reader.initialize(conf, null);
            reader.next();
            reader.reset();
        }
    }

    @Test
    public void testReadAllTypes() throws Exception {
        TestDb.buildAllTypesTable(conn);
        try (JDBCRecordReader reader = new JDBCRecordReader("SELECT * FROM AllTypes", dataSource)) {
            reader.initialize(null);
            List<Writable> item = reader.next();

            assertEquals(item.size(), 15);
            assertEquals(BooleanWritable.class, item.get(0).getClass()); // boolean to boolean
            assertEquals(Text.class, item.get(1).getClass()); // date to text
            assertEquals(Text.class, item.get(2).getClass()); // time to text
            assertEquals(Text.class, item.get(3).getClass()); // timestamp to text
            assertEquals(Text.class, item.get(4).getClass()); // char to text
            assertEquals(Text.class, item.get(5).getClass()); // long varchar to text
            assertEquals(Text.class, item.get(6).getClass()); // varchar to text
            assertEquals(DoubleWritable.class,
                item.get(7).getClass()); // float to double (derby's float is an alias of double by default)
            assertEquals(FloatWritable.class, item.get(8).getClass()); // real to float
            assertEquals(DoubleWritable.class, item.get(9).getClass()); // decimal to double
            assertEquals(DoubleWritable.class, item.get(10).getClass()); // numeric to double
            assertEquals(DoubleWritable.class, item.get(11).getClass()); // double to double
            assertEquals(IntWritable.class, item.get(12).getClass()); // integer to integer
            assertEquals(IntWritable.class, item.get(13).getClass()); // small int to integer
            assertEquals(LongWritable.class, item.get(14).getClass()); // bigint to long

        }
    }

    @Test(expected = RuntimeException.class)
    public void testNextNoMoreShouldFail() throws Exception {
        try (JDBCRecordReader reader = getInitializedReader("SELECT * FROM Coffee")) {
            while (reader.hasNext()) {
                reader.next();
            }
            reader.next();
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidMetadataShouldFail() throws Exception {
        try (JDBCRecordReader reader = getInitializedReader("SELECT * FROM Coffee")) {
            RecordMetaDataLine md = new RecordMetaDataLine(1, new URI("file://test"), JDBCRecordReader.class);
            reader.loadFromMetaData(md);
        }
    }

    private JDBCRecordReader getInitializedReader(String query) throws Exception {
        int[] indices = {1}; // ProdNum column
        JDBCRecordReader reader = new JDBCRecordReader(query, dataSource, "SELECT * FROM Coffee WHERE ProdNum = ?",
            indices);
        reader.setTrimStrings(true);
        reader.initialize(null);
        return reader;
    }
}