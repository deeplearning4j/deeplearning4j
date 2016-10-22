package org.datavec.dataframe.columns;

import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.store.ColumnMetadata;
import org.datavec.dataframe.util.Selection;
import it.unimi.dsi.fastutil.ints.IntComparator;

/**
 * The general interface for columns.
 * <p>
 * Columns can either exist on their own or be a part of a table. All the data in a single column is of a particular
 * type.
 */
public interface Column<E extends Column> {

    int size();

    Table summary();

    default Column subset(Selection rows) {
        Column c = this.emptyCopy();
        for (int row : rows) {
            c.addCell(getString(row));
        }
        return c;
    }

    /**
     * Returns the count of missing values in this column
     */
    int countMissing();

    /**
     * Returns the count of unique values in this column
     */
    int countUnique();

    /**
     * Returns a column of the same type as the receiver, containing only the unique values of the receiver
     */
    Column unique();

    /**
     * Returns the column's name
     */
    String name();

    /**
     * Sets the columns name to the given string
     *
     * @param name The new name MUST be unique for any table containing this column
     */
    void setName(String name);

    /**
     * Returns this column's ColumnType
     */
    ColumnType type();

    /**
     * Returns a string representation of the value at the given row
     */
    String getString(int row);

    /**
     * Returns a copy of the receiver with no data. The column name and type are the same
     */
    Column emptyCopy();

    /**
     * Returns a deep copy of the receiver
     */
    Column copy();

    /**
     * Returns an empty copy of the receiver, with its internal storage initialized to the given row size
     */
    Column emptyCopy(int rowSize);

    void clear();

    void sortAscending();

    void sortDescending();

    /**
     * Returns true if the column has no data
     */
    boolean isEmpty();

    void addCell(String stringValue);

    /**
     * Returns a unique string that identifies this column
     */
    String id();

    /**
     * Returns a String containing the column's metadata in json format
     */
    String metadata();

    ColumnMetadata columnMetadata();

    IntComparator rowComparator();

    default String first() {
        return getString(0);
    }

    default String last() {
        return getString(size() - 1);
    }

    void append(Column column);

    default Column first(int numRows) {
        Column col = emptyCopy();
        int rows = Math.min(numRows, size());
        for (int i = 0; i < rows; i++) {
            col.addCell(getString(i));
        }
        return col;
    }

    default Column last(int numRows) {
        Column col = emptyCopy();
        int rows = Math.min(numRows, size());
        for (int i = size() - rows; i < size(); i++) {
            col.addCell(getString(i));
        }
        return col;
    }

    String print();

    default String title() {
        return "Column: " + name() + '\n';
    }

    String comment();

    void setComment(String comment);

    default double[] toDoubleArray() {
        throw new UnsupportedOperationException("Method toDoubleArray() is not supported on non-numeric columns");
    }

    int columnWidth();

    Selection isMissing();

    Selection isNotMissing();

    /**
     * Returns the width of a cell in this column, in bytes
     */
    int byteSize();

    /**
     * Returns the contents of the cell at rowNumber as a byte[]
     */
    byte[] asBytes(int rowNumber);

    /**
     * Returns a new column of the same type as the receiver, such that the values in the new column
     * contain the difference between each cell in the original and it's predecessor.
     * The Missing Value Indicator is used for the first cell in the new column.
     * (e.g. IntColumn.MISSING_VALUE)
     */
    E difference();
}
