package org.nd4j.jdbc.hsql;

import org.nd4j.jdbc.loader.impl.BaseLoader;

import javax.sql.DataSource;

/**
 * HSQLDB loader for ndarrays.
 *
 * @author Adam Gibson
 */
public class HsqlLoader extends BaseLoader {

    public HsqlLoader(DataSource dataSource, String jdbcUrl, String tableName, String idColumnName, String columnName) throws Exception {
        super(dataSource, jdbcUrl, tableName, idColumnName, columnName);
    }

    public HsqlLoader(String jdbcUrl, String tableName, String idColumnName, String columnName) throws Exception {
        super(jdbcUrl, tableName, idColumnName, columnName);
    }

    public HsqlLoader(DataSource dataSource, String jdbcUrl, String tableName, String columnName) throws Exception {
        super(dataSource, jdbcUrl, tableName, columnName);
    }


    /**
     * Create an insert statement
     *
     * @return a new insert statement
     */
    @Override
    public String insertStatement() {
        return "INSERT INTO " + tableName + " VALUES(?,?)";
    }

    /**
     * Create an insert statement. This should be a templated query.
     * IE: Question mark at the end, we will take care of setting the proper value.
     *
     * @return a new insert statement
     */
    @Override
    public String loadStatement() {
        return "SELECT * FROM " + tableName + " WHERE " + this.idColumnName + " =?";


    }

    /**
     * Create an delete statement
     *
     * @return a new delete statement
     */
    @Override
    public String deleteStatement() {
        return "DELETE  FROM " + tableName + " WHERE " + this.idColumnName + " =?";

    }
}
