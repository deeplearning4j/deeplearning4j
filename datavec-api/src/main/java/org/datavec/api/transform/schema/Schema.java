package org.datavec.api.transform.schema;

import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.metadata.*;
import org.joda.time.DateTimeZone;

import java.io.Serializable;
import java.util.*;

/**
 * Created by Alex on 4/03/2016.
 */
public class Schema implements Serializable {

    private List<String> columnNames;
    private List<ColumnMetaData> columnMetaData;
    private Map<String, Integer> columnNamesIndex;   //For efficient lookup


    protected Schema(Builder builder) {
        this.columnNames = builder.columnNames;
        this.columnMetaData = builder.columnMetaData;
        columnNamesIndex = new HashMap<>();
        for (int i = 0; i < columnNames.size(); i++) {
            columnNamesIndex.put(columnNames.get(i), i);
        }
    }

    public Schema(List<String> columnNames, List<ColumnMetaData> columnMetaData) {
        if (columnNames == null || columnMetaData == null) throw new IllegalArgumentException("Input cannot be null");
        if (columnNames.size() == 0 || columnNames.size() != columnMetaData.size())
            throw new IllegalArgumentException("List sizes must match (and be non-zero)");
        this.columnNames = columnNames;
        this.columnMetaData = columnMetaData;
        this.columnNamesIndex = new HashMap<>();
        for (int i = 0; i < columnNames.size(); i++) {
            columnNamesIndex.put(columnNames.get(i), i);
        }
    }

    public Schema newSchema(List<String> columnNames, List<ColumnMetaData> columnMetaData) {
        return new Schema(columnNames, columnMetaData);
    }

    public int numColumns() {
        return columnNames.size();
    }

    public String getName(int column) {
        return columnNames.get(column);
    }

    public ColumnType getType(int column) {
        return columnMetaData.get(column).getColumnType();
    }

    public ColumnMetaData getMetaData(int column) {
        return columnMetaData.get(column);
    }

    public ColumnMetaData getMetaData(String column) {
        return getMetaData(getIndexOfColumn(column));
    }

    public List<String> getColumnNames() {
        return new ArrayList<>(columnNames);
    }

    public List<ColumnType> getColumnTypes() {
        List<ColumnType> list = new ArrayList<>(columnMetaData.size());
        for (ColumnMetaData md : columnMetaData) list.add(md.getColumnType());
        return list;
    }

    public List<ColumnMetaData> getColumnMetaData() {
        return new ArrayList<>(columnMetaData);
    }

    public int getIndexOfColumn(String columnName) {
        Integer idx = columnNamesIndex.get(columnName);
        if (idx == null) throw new NoSuchElementException("Unknown column: \"" + columnName + "\"");
        return idx;
    }

    /**
     * Determine if the schema has a column with the specified name
     *
     * @param columnName Name to see if the column exists
     * @return True if column exists for that name, false otherwise
     */
    public boolean hasColumn(String columnName) {
        Integer idx = columnNamesIndex.get(columnName);
        return idx != null;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        int nCol = numColumns();

        int maxNameLength = 0;
        for (String s : getColumnNames()) {
            maxNameLength = Math.max(maxNameLength, s.length());
        }

        //Header:
        sb.append("Schema():\n");
        sb.append(String.format("%-6s", "idx")).append(String.format("%-" + (maxNameLength + 8) + "s", "name"))
                .append(String.format("%-15s", "type")).append("meta data").append("\n");

        for (int i = 0; i < nCol; i++) {
            String colName = getName(i);
            ColumnType type = getType(i);
            ColumnMetaData meta = getMetaData(i);
            String paddedName = String.format("%-" + (maxNameLength + 8) + "s", "\"" + colName + "\"");
            sb.append(String.format("%-6d", i))
                    .append(paddedName)
                    .append(String.format("%-15s", type))
                    .append(meta).append("\n");
        }

        return sb.toString();
    }

    public static class Builder {

        List<String> columnNames = new ArrayList<>();
        List<ColumnMetaData> columnMetaData = new ArrayList<>();

        /**
         * Add a Double column with no restrictions on the allowable values, except for no NaN/infinite values allowed
         *
         * @param name Name of the column
         */
        public Builder addColumnDouble(String name) {
            return addColumn(name, new DoubleMetaData());
        }

        /**
         * Add a Double column with the specified restrictions (and no NaN/Infinite values allowed)
         *
         * @param name            Name of the column
         * @param minAllowedValue Minimum allowed value (inclusive). If null: no restriction
         * @param maxAllowedValue Maximum allowed value (inclusive). If null: no restriction
         * @return
         */
        public Builder addColumnDouble(String name, Double minAllowedValue, Double maxAllowedValue) {
            return addColumnDouble(name, minAllowedValue, maxAllowedValue, false, false);
        }

        /**
         * Add a double column with the specified restrictions
         *
         * @param name            Name of the column
         * @param minAllowedValue Minimum allowed value (inclusive). If null: no restriction
         * @param maxAllowedValue Maximum allowed value (inclusive). If null: no restriction
         * @param allowNaN        If false: don't allow NaN values. If true: allow.
         * @param allowInfinite   If false: don't allow infinite values. If true: allow
         */
        public Builder addColumnDouble(String name, Double minAllowedValue, Double maxAllowedValue,
                                       boolean allowNaN, boolean allowInfinite) {
            return addColumn(name, new DoubleMetaData(minAllowedValue, maxAllowedValue, allowNaN, allowInfinite));
        }

        /**
         * Add multiple columns with no restrictions on the allowable values of the columns (other than no NaN/Infinite)
         *
         * @param columnNames Names of the columns to add
         */
        public Builder addColumnsDouble(String... columnNames) {
            for (String s : columnNames) addColumnDouble(s);
            return this;
        }

        /**
         * Add an integer column with no restrictions on the allowable values
         *
         * @param name Name of the column
         */
        public Builder addColumnInteger(String name) {
            return addColumn(name, new IntegerMetaData());
        }

        /**
         * Add an integer column with the specified min/max allowable values
         *
         * @param name            Name of the column
         * @param minAllowedValue Minimum allowed value (inclusive). If null: no restriction
         * @param maxAllowedValue Maximum allowed value (inclusive). If null: no restriction
         */
        public Builder addColumnInteger(String name, Integer minAllowedValue, Integer maxAllowedValue) {
            return addColumn(name, new IntegerMetaData(minAllowedValue, maxAllowedValue));
        }

        /**
         * Add multiple Integer columns with no restrictions on the min/max allowable values
         *
         * @param names Names of the integer columns to add
         */
        public Builder addColumnsInteger(String... names) {
            for (String s : names) addColumnInteger(s);
            return this;
        }

        /**
         * Add a Categorical column, with the specified state names
         *
         * @param name       Name of the column
         * @param stateNames Names of the allowable states for this categorical column
         */
        public Builder addColumnCategorical(String name, String... stateNames) {
            return addColumn(name, new CategoricalMetaData(stateNames));
        }

        /**
         * Add a Categorical column, with the specified state names
         *
         * @param name       Name of the column
         * @param stateNames Names of the allowable states for this categorical column
         */
        public Builder addColumnCategorical(String name, List<String> stateNames) {
            return addColumn(name, new CategoricalMetaData(stateNames));
        }

        /**
         * Add a Long column, with no restrictions on the min/max values
         *
         * @param name Name of the column
         */
        public Builder addColumnLong(String name) {
            return addColumn(name, new LongMetaData());
        }

        /**
         * Add an Long column with the specified min/max allowable values
         *
         * @param name            Name of the column
         * @param minAllowedValue Minimum allowed value (inclusive). If null: no restriction
         * @param maxAllowedValue Maximum allowed value (inclusive). If null: no restriction
         */
        public Builder addColumnLong(String name, Long minAllowedValue, Long maxAllowedValue) {
            return addColumn(name, new LongMetaData(minAllowedValue, maxAllowedValue));
        }

        /**
         * Add multiple long columns, with no restrictions on the allowable values
         *
         * @param names Names of the Long columns to add
         */
        public Builder addColumnsLong(String... names) {
            for (String s : names) addColumnLong(s);
            return this;
        }

        /**
         * Add a column
         *
         * @param name     Name of the column
         * @param metaData metadata for this column
         */
        public Builder addColumn(String name, ColumnMetaData metaData) {
            columnNames.add(name);
            columnMetaData.add(metaData);
            return this;
        }

        /**
         * Add a String column with no restrictions on the allowable values.
         *
         * @param name Name of  the column
         */
        public Builder addColumnString(String name) {
            return addColumn(name, new StringMetaData());
        }

        /**
         * Add String columns with no restrictions on the allowable values
         *
         * @param columnNames Names of the String columns to add
         */
        public Builder addColumnsString(String... columnNames) {
            for (String s : columnNames) addColumnString(s);
            return this;
        }

        /**
         * Add a String column with the specified restrictions
         *
         * @param name               Name of the column
         * @param regex              Regex that the String must match in order to be considered value. If null: no regex restriction
         * @param minAllowableLength Minimum allowable length for the String to be considered valid
         * @param maxAllowableLength Maximum allowable length for the String to be considered valid
         */
        public Builder addColumnString(String name, String regex, Integer minAllowableLength, int maxAllowableLength) {
            return addColumn(name, new StringMetaData(regex, minAllowableLength, maxAllowableLength));
        }

        /**
         * Add a Time column with no restrictions on the min/max allowable times
         * <b>NOTE</b>: Time columns are represented by LONG (epoch millisecond) values. For time values in human-readable formats,
         * use String columns + StringToTimeTransform
         *
         * @param columnName Name of the column
         * @param timeZone   Time zone of the time column
         */
        public Builder addColumnTime(String columnName, TimeZone timeZone) {
            return addColumnTime(columnName, DateTimeZone.forTimeZone(timeZone));
        }

        /**
         * Add a Time column with no restrictions on the min/max allowable times
         * <b>NOTE</b>: Time columns are represented by LONG (epoch millisecond) values. For time values in human-readable formats,
         * use String columns + StringToTimeTransform
         *
         * @param columnName Name of the column
         * @param timeZone   Time zone of the time column
         */
        public Builder addColumnTime(String columnName, DateTimeZone timeZone) {
            return addColumnTime(columnName, timeZone, null, null);
        }

        /**
         * Add a Time column with the specified restrictions
         * <b>NOTE</b>: Time columns are represented by LONG (epoch millisecond) values. For time values in human-readable formats,
         * use String columns + StringToTimeTransform
         *
         * @param columnName    Name of the column
         * @param timeZone      Time zone of the time column
         * @param minValidValue Minumum allowable time (in milliseconds). May be null.
         * @param maxValidValue Maximum allowable time (in milliseconds). May be null.
         */
        public Builder addColumnTime(String columnName, DateTimeZone timeZone, Long minValidValue, Long maxValidValue) {
            addColumn(columnName, new TimeMetaData(timeZone, minValidValue, maxValidValue));
            return this;
        }

        /**
         * Create the Schema
         */
        public Schema build() {
            return new Schema(this);
        }
    }

}
