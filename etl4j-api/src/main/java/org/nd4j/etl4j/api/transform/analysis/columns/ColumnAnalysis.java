package org.nd4j.etl4j.api.transform.analysis.columns;

import io.skymind.echidna.api.ColumnType;

import java.io.Serializable;

/**
 * Interface for column analysis
 */
public interface ColumnAnalysis extends Serializable {

    long getCountTotal();

    ColumnType getColumnType();

}
