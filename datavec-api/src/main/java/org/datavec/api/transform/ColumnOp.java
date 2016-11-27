package org.datavec.api.transform;

import org.datavec.api.transform.schema.Schema;


/**
 * ColumnOp
 * is a transform meant
 * to run over 1 or more columns
 *
 * @author Adam Gibson
 */
public interface ColumnOp {
    /** Get the output schema for this transformation, given an input schema */
    Schema transform(Schema inputSchema);


    /** Set the input schema.
     */
    void setInputSchema(Schema inputSchema);

    /**
     * Getter for input schema
     * @return
     */
    Schema getInputSchema();

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
