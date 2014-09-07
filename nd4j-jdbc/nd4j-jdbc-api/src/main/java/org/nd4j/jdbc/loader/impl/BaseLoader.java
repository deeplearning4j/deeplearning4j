package org.nd4j.jdbc.loader.impl;

import com.mchange.v2.c3p0.ComboPooledDataSource;
import org.nd4j.jdbc.driverfinder.DriverFinder;
import org.nd4j.jdbc.loader.api.JDBCNDArrayIO;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.sql.DataSource;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.sql.*;

/**
 * Base class for loading ndarrays via org.nd4j.jdbc
 *
 * @author Adam Gibson
 */

public abstract class BaseLoader implements JDBCNDArrayIO {

    protected String tableName,columnName,idColumnName,jdbcUrl;
    protected DataSource dataSource;

    protected BaseLoader(String jdbcUrl,String tableName, String idColumnName,String columnName) throws Exception {
        this.jdbcUrl = jdbcUrl;
        this.tableName = tableName;
        this.columnName = columnName;
        dataSource   = new ComboPooledDataSource();
        ComboPooledDataSource c = (ComboPooledDataSource) dataSource;
        c.setJdbcUrl(jdbcUrl);
        c.setDriverClass(DriverFinder.getDriver().getClass().getName());
        this.idColumnName = idColumnName;

    }

    protected BaseLoader(String jdbcUrl,String tableName,String columnName) throws Exception {
        this(jdbcUrl,tableName,"id",columnName);

    }

    /**
     * Load an ndarray from a blob
     *
     * @param blob the blob to load from
     * @return the loaded ndarray
     */
    @Override
    public INDArray load(Blob blob) throws SQLException, IOException {
        if(blob == null)
            return null;
        DataInputStream dis = new DataInputStream(blob.getBinaryStream());
        return Nd4j.read(dis);
    }

    /**
     * Load a complex ndarray from a blob
     *
     * @param blob the blob to load from
     * @return the complex ndarray
     */
    @Override
    public IComplexNDArray loadComplex(Blob blob) throws SQLException, IOException {
        DataInputStream dis = new DataInputStream(blob.getBinaryStream());
        return Nd4j.readComplex(dis);
    }

    /**
     * Save the ndarray
     *
     * @param save the ndarray to save
     */
    @Override
    public void save(INDArray save,String id) throws SQLException, IOException {
        doSave(save,id);

    }

    /**
     * Save the ndarray
     *
     * @param save the ndarray to save
     */
    @Override
    public void save(IComplexNDArray save,String id) throws IOException, SQLException {
        doSave(save,id);
    }


    private void doSave(INDArray save,String id) throws SQLException, IOException {
        Connection c = dataSource.getConnection();
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        if(save instanceof IComplexNDArray) {
            IComplexNDArray c2 = (IComplexNDArray) save;
            Nd4j.writeComplex(c2,dos);
        }

        else
            Nd4j.write(save,dos);

        byte[] bytes = bos.toByteArray();

        PreparedStatement preparedStatement = c.prepareStatement(insertStatement());
        preparedStatement.setString(1,id);
        preparedStatement.setBytes(2,bytes);
        int update = preparedStatement.executeUpdate();
        preparedStatement.close();
        c.close();



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
        preparedStatement.setString(1,id);
        ResultSet r = preparedStatement.executeQuery();
        if(r.wasNull() || !r.next())
            return null;
        else {
            Blob first = r.getBlob(2);
            c.close();
            r.close();
            return first;
        }
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
        p.setString(1,id);
        p.execute();

    }
}
