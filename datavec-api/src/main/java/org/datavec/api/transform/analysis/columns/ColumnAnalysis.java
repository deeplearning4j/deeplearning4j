package org.datavec.api.transform.analysis.columns;

import org.datavec.api.transform.ColumnType;

import java.io.Serializable;

/**
 * Interface for column analysis
 */
public interface ColumnAnalysis extends Serializable {

    long getCountTotal();

    ColumnType getColumnType();

}
