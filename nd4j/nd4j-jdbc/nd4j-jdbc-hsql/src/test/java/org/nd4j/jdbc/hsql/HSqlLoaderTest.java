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

package org.nd4j.jdbc.hsql;

import org.hsqldb.jdbc.JDBCDataSource;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.sql.DataSource;
import java.sql.*;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThat;

public class HSqlLoaderTest {
    private static HsqlLoader hsqlLoader;
    private static DataSource dataSource;

    public final static String JDBC_URL = "jdbc:hsqldb:mem:ndarrays";
    public final static String TABLE_NAME = "testarrays";
    public final static String ID_COLUMN_NAME = "id";
    public final static String COLUMN_NAME = "array";

    @BeforeClass
    public static void init() throws Exception  {
        hsqlLoader = new HsqlLoader(dataSource(),JDBC_URL,TABLE_NAME,ID_COLUMN_NAME,COLUMN_NAME);
        Class.forName("org.hsqldb.jdbc.JDBCDriver");

        // initialize database
        initDatabase();
    }


    public static DataSource dataSource() {
        if (dataSource != null)
            return dataSource;
        JDBCDataSource dataSource = new JDBCDataSource();
        dataSource.setDatabase(JDBC_URL);
        dataSource.setUrl(JDBC_URL);
        dataSource.setPassword("test");
        dataSource.setUser("test");
        HSqlLoaderTest.dataSource = dataSource;
        return dataSource;
    }



    @AfterClass
    public static void destroy() throws SQLException {
        try (Connection connection = getConnection(); Statement statement = connection.createStatement()) {
            statement.executeUpdate("DROP TABLE " + TABLE_NAME);
            connection.commit();
        }
    }

    /**
     * Database initialization for testing i.e.
     * <ul>
     * <li>Creating Table</li>
     * <li>Inserting record</li>
     * </ul>
     *
     * @throws SQLException
     */
    private static void initDatabase() throws Exception {
        try (Connection connection = getConnection(); Statement statement = connection.createStatement()) {
            statement.execute(String.format("CREATE TABLE %s (%s INT NOT NULL,"
                    + " %s BLOB NOT NULL, PRIMARY KEY (id))",TABLE_NAME,ID_COLUMN_NAME,COLUMN_NAME));
            connection.commit();
            hsqlLoader.save(Nd4j.linspace(1,4,4),"1");
            connection.commit();
        }
    }

    /**
     * Create a connection
     *
     * @return connection object
     * @throws SQLException
     */
    private static Connection getConnection() throws SQLException {
        return DriverManager.getConnection(JDBC_URL, "test", "test");
    }

    /**
     * Get total records in table
     *
     * @return total number of records. In case of exception 0 is returned
     */
    private int getTotalRecords() {
        try (Connection connection = getConnection(); Statement statement = connection.createStatement()) {
            ResultSet result = statement.executeQuery(String.format("SELECT count(*) as total FROM %s",TABLE_NAME));
            if (result.next()) {
                return result.getInt("total");
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return 0;
    }

    @Test
    public void getTotalRecordsTest() throws Exception {
        assertThat(1, is(getTotalRecords()));

        INDArray load = hsqlLoader.load(hsqlLoader.loadForID("1"));
        assertNotNull(load);
        assertEquals(Nd4j.linspace(1,4,4),load);


    }


}
