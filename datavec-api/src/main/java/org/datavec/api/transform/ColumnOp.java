package org.datavec.api.transform;

/**
 * ColumnOp
 * is a transform meant
 * to run over 1 or more columns
 *
 * @author Adam Gibson
 */
public interface ColumnOp {


    /**
     * The output column name
     * after the operation has been applied
     * @return the output column name
     */
    String outputColumnName();

    /**
     * The output column names
     * This will often be the same as the input
     * @return the output column names
     */
    String[] outputColumnNames();

    /**
     * Returns column names
     * this op is meant to run on
     * @return
     */
    String[] columnNames();

    /**
     * Returns a singular column name
     * this op is meant to run on
     * @return
     */
    String columnName();

}
