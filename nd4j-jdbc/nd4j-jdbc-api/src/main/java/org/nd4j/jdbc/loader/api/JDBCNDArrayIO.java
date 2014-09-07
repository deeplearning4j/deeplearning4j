package org.nd4j.jdbc.loader.api;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.sql.Blob;
import java.sql.SQLException;

/**
 * Load a complex ndarray via org.nd4j.jdbc
 *
 * @author Adam Gibson
 */
public interface JDBCNDArrayIO {

    /**
     * Load an ndarray from a blob
     * @param blob the blob to load from
     * @return the loaded ndarray
     */
    INDArray load(Blob blob) throws IOException, SQLException;

    /**
     * Load a complex ndarray from a blob
     * @param blob the blob to load from
     * @return the complex ndarray
     */
    IComplexNDArray loadComplex(Blob blob) throws IOException, SQLException;


    /**
     * Create an insert statement
     * @return a new insert statement
     */
    String insertStatement();

    /**
     * Create an insert statement
     * @return a new insert statement
     */
    String loadStatement();


    /**
     * Create an insert statement
     * @return a new insert statement
     */
    String deleteStatement();

    /**
     * Save the ndarray
     * @param save the ndarray to save
     */
    void save(INDArray save, String id) throws SQLException, IOException;

    /**
     * Save the ndarray
     * @param save the ndarray to save
     */
    void save(IComplexNDArray save, String id) throws IOException, SQLException;

    /**
     * Load an ndarray blob given an id
     * @param id the id to load
     * @return the blob
     */
    Blob  loadForID(String id) throws SQLException;

    /**
     * Delete the given ndarray
     * @param id the id of the ndarray to delete
     */
    void delete(String id) throws SQLException;


}
