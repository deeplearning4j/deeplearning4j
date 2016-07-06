package org.nd4j.etl4j.api.transform;

/**
 * The type of column.
 */
public enum ColumnType {
    String,
    Integer,
    Long,
    Double,
    Categorical,
    Time,
    Bytes    //Arbitrary byte[] data

}
