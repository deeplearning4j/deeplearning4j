package org.nd4j.etl4j.api.transform.analysis.columns;

import org.nd4j.etl4j.api.transform.ColumnType;

import java.io.Serializable;

/**
 * Interface for column analysis
 */
public interface ColumnAnalysis extends Serializable {

    long getCountTotal();

    ColumnType getColumnType();

}
