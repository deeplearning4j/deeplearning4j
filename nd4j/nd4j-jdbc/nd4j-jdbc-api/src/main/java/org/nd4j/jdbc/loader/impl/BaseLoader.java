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

package org.nd4j.jdbc.loader.impl;

import com.mchange.v2.c3p0.ComboPooledDataSource;
import org.nd4j.jdbc.driverfinder.DriverFinder;
import org.nd4j.jdbc.loader.api.JDBCNDArrayIO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.binary.BinarySerde;

import javax.sql.DataSource;
import java.io.*;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.sql.*;

/**
 * Base class for loading ndarrays via org.nd4j.jdbc
 *
 * @author Adam Gibson
 */

public abstract class BaseLoader implements JDBCNDArrayIO {

    protected String tableName, columnName, idColumnName, jdbcUrl;
    protected DataSource dataSource;

    protected BaseLoader(DataSource dataSource, String jdbcUrl, String tableName, String idColumnName,
                         String columnName) throws Exception {
        this.dataSource = dataSource;
        this.jdbcUrl = jdbcUrl;
        this.tableName = tableName;
        this.columnName = columnName;
        this.idColumnName = idColumnName;
        if (dataSource == null) {
            dataSource = new ComboPooledDataSource();
            ComboPooledDataSource c = (ComboPooledDataSource) dataSource;
            c.setJdbcUrl(jdbcUrl);
            c.setDriverClass(DriverFinder.getDriver().getClass().getName());

        }
    }


    protected BaseLoader(String jdbcUrl, String tableName, String idColumnName, String columnName) throws Exception {
        this.jdbcUrl = jdbcUrl;
        this.tableName = tableName;
        this.columnName = columnName;
        dataSource = new ComboPooledDataSource();
        ComboPooledDataSource c = (ComboPooledDataSource) dataSource;
        c.setJdbcUrl(jdbcUrl);
        c.setDriverClass(DriverFinder.getDriver().getClass().getName());
        this.idColumnName = idColumnName;

    }

    protected BaseLoader(DataSource dataSource, String jdbcUrl, String tableName, String columnName) throws Exception {
        this(dataSource, jdbcUrl, tableName, "id", columnName);

    }

    /**
     * Convert an ndarray to a blob
     *
     * @param toConvert the ndarray to convert
     * @return the converted ndarray
     */
    @Override
    public Blob convert(INDArray toConvert) throws SQLException {
        ByteBuffer byteBuffer = BinarySerde.toByteBuffer(toConvert);
        Buffer buffer = (Buffer) byteBuffer;
        buffer.rewind();
        byte[] arr = new byte[byteBuffer.capacity()];
        byteBuffer.get(arr);
        Connection c = dataSource.getConnection();
        Blob b = c.createBlob();
        b.setBytes(1, arr);
        return b;
    }

    /**
     * Load an ndarray from a blob
     *
     * @param blob the blob to load from
     * @return the loaded ndarray
     */
    @Override
    public INDArray load(Blob blob) throws SQLException {
        if (blob == null)
            return null;
        try(InputStream is = blob.getBinaryStream()) {
            ByteBuffer direct = ByteBuffer.allocateDirect((int) blob.length());
            ReadableByteChannel readableByteChannel = Channels.newChannel(is);
            readableByteChannel.read(direct);
            Buffer byteBuffer = (Buffer) direct;
            byteBuffer.rewind();
            return BinarySerde.toArray(direct);
        } catch (Exception e) {
           throw new RuntimeException(e);
        }


    }

    /**
     * Save the ndarray
     *
     * @param save the ndarray to save
     */
    @Override
    public void save(INDArray save, String id) throws SQLException, IOException {
        doSave(save, id);

    }


    private void doSave(INDArray save, String id) throws SQLException, IOException {
        Connection c = dataSource.getConnection();
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        BinarySerde.writeArrayToOutputStream(save,bos);

        byte[] bytes = bos.toByteArray();

        PreparedStatement preparedStatement = c.prepareStatement(insertStatement());
        preparedStatement.setString(1, id);
        preparedStatement.setBytes(2, bytes);
        preparedStatement.executeUpdate();


    }


    /**
     * Load an ndarray blob given an id
     *
     * @param id the id to load
     * @return the blob
     */
    @Override
    public Blob loadForID(String id) throws SQLException {
        Connection c = dataSource.getConnection();
        PreparedStatement preparedStatement = c.prepareStatement(loadStatement());
        preparedStatement.setString(1, id);
        ResultSet r = preparedStatement.executeQuery();
        if (r.wasNull() || !r.next()) {
            return null;
        } else {
            Blob first = r.getBlob(2);
            return first;
        }


    }

    @Override
    public INDArray loadArrayForId(String id) throws SQLException {
        return load(loadForID(id));
    }

    /**
     * Delete the given ndarray
     *
     * @param id the id of the ndarray to delete
     */
    @Override
    public void delete(String id) throws SQLException {
        Connection c = dataSource.getConnection();
        PreparedStatement p = c.prepareStatement(deleteStatement());
        p.setString(1, id);
        p.execute();


    }
}
