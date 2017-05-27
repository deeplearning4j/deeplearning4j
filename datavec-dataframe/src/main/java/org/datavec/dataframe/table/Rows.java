package org.datavec.dataframe.table;

import org.datavec.dataframe.api.BooleanColumn;
import org.datavec.dataframe.api.CategoryColumn;
import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.api.DateColumn;
import org.datavec.dataframe.api.DateTimeColumn;
import org.datavec.dataframe.api.FloatColumn;
import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.api.LongColumn;
import org.datavec.dataframe.api.ShortColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.api.TimeColumn;
import org.datavec.dataframe.util.Selection;
import it.unimi.dsi.fastutil.ints.IntArrayList;

import net.jcip.annotations.Immutable;

/**
 * A static utility class for row operations
 */
@Immutable
public class Rows {

    // Don't instantiate
    private Rows() {}

    /**
     * Copy the given rows to a table
     * @param rows
     * @param oldTable
     * @param newTable
     */
    public static void copyRowsToTable(IntArrayList rows, Table oldTable, Table newTable) {

        for (int columnIndex = 0; columnIndex < oldTable.columnCount(); columnIndex++) {
            ColumnType columnType = oldTable.column(columnIndex).type();
            switch (columnType) {
                case FLOAT:
                    copy(rows, (FloatColumn) oldTable.column(columnIndex), (FloatColumn) newTable.column(columnIndex));
                    break;
                case INTEGER:
                    copy(rows, (IntColumn) oldTable.column(columnIndex), (IntColumn) newTable.column(columnIndex));
                    break;
                case SHORT_INT:
                    copy(rows, (ShortColumn) oldTable.column(columnIndex), (ShortColumn) newTable.column(columnIndex));
                    break;
                case LONG_INT:
                    copy(rows, (LongColumn) oldTable.column(columnIndex), (LongColumn) newTable.column(columnIndex));
                    break;
                case CATEGORY:
                    copy(rows, (CategoryColumn) oldTable.column(columnIndex),
                                    (CategoryColumn) newTable.column(columnIndex));
                    break;
                case BOOLEAN:
                    copy(rows, (BooleanColumn) oldTable.column(columnIndex),
                                    (BooleanColumn) newTable.column(columnIndex));
                    break;
                case LOCAL_DATE:
                    copy(rows, (DateColumn) oldTable.column(columnIndex), (DateColumn) newTable.column(columnIndex));
                    break;
                case LOCAL_DATE_TIME:
                    copy(rows, (DateTimeColumn) oldTable.column(columnIndex),
                                    (DateTimeColumn) newTable.column(columnIndex));
                    break;
                case LOCAL_TIME:
                    copy(rows, (TimeColumn) oldTable.column(columnIndex), (TimeColumn) newTable.column(columnIndex));
                    break;
                default:
                    throw new RuntimeException("Unhandled column type in case statement");
            }
        }
    }

    public static void appendRowToTable(int row, Table oldTable, Table newTable) {

        IntArrayList rows = new IntArrayList();
        rows.add(row);
        copyRowsToTable(rows, oldTable, newTable);
    }

    public static boolean compareRows(int rowInOriginal, Table original, Table tempTable) {

        boolean result;
        for (int columnIndex = 0; columnIndex < original.columnCount(); columnIndex++) {
            ColumnType columnType = original.column(columnIndex).type();
            switch (columnType) {
                case FLOAT:
                    result = compare(rowInOriginal, (FloatColumn) tempTable.column(columnIndex),
                                    (FloatColumn) original.column(columnIndex));
                    if (!result)
                        return false;
                    break;
                case INTEGER:
                    result = compare(rowInOriginal, (IntColumn) tempTable.column(columnIndex),
                                    (IntColumn) original.column(columnIndex));
                    if (!result)
                        return false;
                    break;
                case SHORT_INT:
                    result = compare(rowInOriginal, (ShortColumn) tempTable.column(columnIndex),
                                    (ShortColumn) original.column(columnIndex));
                    if (!result)
                        return false;
                    break;
                case LONG_INT:
                    result = compare(rowInOriginal, (LongColumn) tempTable.column(columnIndex),
                                    (LongColumn) original.column(columnIndex));
                    if (!result)
                        return false;
                    break;
                case CATEGORY:
                    result = compare(rowInOriginal, (CategoryColumn) tempTable.column(columnIndex),
                                    (CategoryColumn) original.column(columnIndex));
                    if (!result)
                        return false;
                    break;
                case BOOLEAN:
                    result = compare(rowInOriginal, (BooleanColumn) tempTable.column(columnIndex),
                                    (BooleanColumn) original.column(columnIndex));
                    if (!result)
                        return false;
                    break;
                case LOCAL_DATE:
                    result = compare(rowInOriginal, (DateColumn) tempTable.column(columnIndex),
                                    (DateColumn) original.column(columnIndex));
                    if (!result)
                        return false;
                    break;
                case LOCAL_DATE_TIME:
                    result = compare(rowInOriginal, (DateTimeColumn) tempTable.column(columnIndex),
                                    (DateTimeColumn) original.column(columnIndex));
                    if (!result)
                        return false;
                    break;
                case LOCAL_TIME:
                    result = compare(rowInOriginal, (TimeColumn) tempTable.column(columnIndex),
                                    (TimeColumn) original.column(columnIndex));
                    if (!result)
                        return false;
                    break;
                default:
                    throw new RuntimeException("Unhandled column type in case statement");
            }
        }
        return true;
    }

    public static void copyRowsToTable(Selection rows, Table oldTable, Table newTable) {
        int[] r = rows.toArray();
        IntArrayList rowArray = new IntArrayList(r);
        copyRowsToTable(rowArray, oldTable, newTable);
    }

    public static void head(int rowCount, Table oldTable, Table newTable) {
        IntArrayList rows = new IntArrayList(rowCount);
        for (int i = 0; i < rowCount; i++) {
            rows.add(i);
        }
        copyRowsToTable(rows, oldTable, newTable);
    }

    public static void tail(int rowsToInclude, Table oldTable, Table newTable) {
        int oldTableSize = oldTable.rowCount();
        int start = oldTableSize - rowsToInclude;
        IntArrayList rows = new IntArrayList(rowsToInclude);
        for (int i = start; i < oldTableSize; i++) {
            rows.add(i);
        }
        copyRowsToTable(rows, oldTable, newTable);
    }

    private static void copy(IntArrayList rows, FloatColumn oldColumn, FloatColumn newColumn) {
        for (int index : rows) {
            newColumn.add(oldColumn.get(index));
        }
    }

    private static boolean compare(int row, FloatColumn tempTable, FloatColumn original) {
        return original.get(row) == tempTable.get(tempTable.size() - 1);
    }

    private static void copy(IntArrayList rows, CategoryColumn oldColumn, CategoryColumn newColumn) {
        newColumn.initializeWith(oldColumn.getValues(rows), oldColumn.dictionaryMap());
    }

    private static void copyRow(int row, CategoryColumn oldColumn, CategoryColumn newColumn) {
        newColumn.add(oldColumn.get(row));
    }

    private static boolean compare(int row, CategoryColumn tempTable, CategoryColumn original) {
        String t = tempTable.get(tempTable.size() - 1);
        String o = original.get(row);
        return o.equals(t);
    }

    private static void copy(IntArrayList rows, BooleanColumn oldColumn, BooleanColumn newColumn) {
        for (int index : rows) {
            newColumn.add(oldColumn.get(index));
        }
    }

    private static boolean compare(int row, BooleanColumn tempTable, BooleanColumn original) {
        return original.get(row) == tempTable.get(tempTable.size() - 1);
    }

    private static void copy(IntArrayList rows, IntColumn oldColumn, IntColumn newColumn) {
        for (int index : rows) {
            newColumn.add(oldColumn.get(index));
        }
    }

    private static boolean compare(int row, IntColumn tempTable, IntColumn original) {
        return original.get(row) == tempTable.get(tempTable.size() - 1);
    }

    private static void copy(IntArrayList rows, ShortColumn oldColumn, ShortColumn newColumn) {
        for (int index : rows) {
            newColumn.add(oldColumn.get(index));
        }
    }

    private static boolean compare(int row, ShortColumn tempTable, ShortColumn original) {
        short t = tempTable.get(tempTable.size() - 1);
        short o = original.get(row);
        return o == t;
    }

    private static void copy(IntArrayList rows, LongColumn oldColumn, LongColumn newColumn) {
        for (int index : rows) {
            newColumn.add(oldColumn.get(index));
        }
    }

    private static boolean compare(int row, LongColumn tempTable, LongColumn original) {
        return original.get(row) == tempTable.get(tempTable.size() - 1);
    }

    private static void copy(IntArrayList rows, DateTimeColumn oldColumn, DateTimeColumn newColumn) {
        for (int index : rows) {
            newColumn.add(oldColumn.getLong(index));
        }
    }

    private static boolean compare(int row, DateTimeColumn tempTable, DateTimeColumn original) {
        return original.getLong(row) == tempTable.getLong(tempTable.size() - 1);
    }

    private static void copy(IntArrayList rows, DateColumn oldColumn, DateColumn newColumn) {
        for (int index : rows) {
            newColumn.add(oldColumn.getInt(index));
        }
    }

    private static boolean compare(int row, DateColumn tempTable, DateColumn original) {
        return original.getInt(row) == tempTable.getInt(tempTable.size() - 1);
    }

    private static void copy(IntArrayList rows, TimeColumn oldColumn, TimeColumn newColumn) {
        for (int index : rows) {
            newColumn.add(oldColumn.getInt(index));
        }
    }

    private static boolean compare(int row, TimeColumn tempTable, TimeColumn original) {
        return original.getInt(row) == tempTable.getInt(tempTable.size() - 1);
    }
}
