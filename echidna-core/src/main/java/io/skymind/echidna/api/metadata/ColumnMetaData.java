package io.skymind.echidna.api.metadata;

import io.skymind.echidna.api.ColumnType;
import org.canova.api.writable.Writable;

import java.io.Serializable;

/**
 * ColumnMetaData: metadata for each column. Used to define:
 * (a) the type of each column, and
 * (b) any restrictions on the allowable values in each column
 *
 * @author Alex Black
 */
public interface ColumnMetaData extends Serializable, Cloneable {

    /**
     * Get the type of column
     */
    ColumnType getColumnType();

    /**
     * Is the given Writable valid for this column, given the column type and any restrictions given by the
     * ColumnMetaData object?
     *
     * @param writable Writable to check
     * @return true if value, false if invalid
     */
    boolean isValid(Writable writable);

    ColumnMetaData clone();
}
