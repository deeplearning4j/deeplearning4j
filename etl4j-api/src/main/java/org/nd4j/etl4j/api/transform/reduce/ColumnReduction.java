package org.nd4j.etl4j.api.transform.reduce;

import org.canova.api.writable.Writable;
import org.nd4j.etl4j.api.transform.metadata.ColumnMetaData;

import java.io.Serializable;
import java.util.List;

/**
 * A column reduction defines how a single column should be reduced.
 * Used in conjunction with {@link Reducer} to provide custom reduction functionality.
 *
 * @author Alex Black
 */
public interface ColumnReduction extends Serializable {

    /**
     * Reduce a single column.
     * <b>Note</b>: The {@code List<Writable>} here is a single <b>column</b> in a reduction window, and NOT the single row
     * (as is usually the case for {@code List<Writable>} instances
     *
     * @param columnData The Writable objects for a column
     * @return Writable containing the reduced data
     */
    Writable reduceColumn(List<Writable> columnData);

    /**
     * Post-reduce: what is the name of the column?
     * For example, "myColumn" -> "mean(myColumn)"
     *
     * @param columnInputName Name of the column before reduction
     * @return Name of the column after the reduction
     */
    String getColumnOutputName(String columnInputName);

    /**
     * Post-reduce: what is the metadata (type, etc) for this column?
     * For example: a "count unique" operation on a String (StringMetaData) column would return an Integer (IntegerMetaData) column
     *
     * @param columnInputMeta Metadata for the column, before reduce
     * @return Metadata for the column, after the reduction
     */
    ColumnMetaData getColumnOutputMetaData(ColumnMetaData columnInputMeta);

}
