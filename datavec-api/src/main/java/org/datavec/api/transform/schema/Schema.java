/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.transform.schema;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.metadata.*;
import org.datavec.api.transform.serde.JsonMappers;
import org.datavec.api.transform.serde.legacy.LegacyMappingHelper;
import org.datavec.api.writable.*;
import org.joda.time.DateTimeZone;
import org.nd4j.shade.jackson.annotation.*;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;
import org.nd4j.shade.jackson.datatype.joda.JodaModule;

import java.io.Serializable;
import java.util.*;

/**
 * A Schema defines the layout of tabular data. Specifically, it contains names f
 * or each column, as well as details of types
 * (Integer, String, Long, Double, etc).<br>
 * Type information for each column may optionally include
 * restrictions on the allowable values for each column.<br>
 * <p>
 * See also: {@link SequenceSchema}
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"columnNames", "columnNamesIndex"})
@EqualsAndHashCode
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class",
        defaultImpl = LegacyMappingHelper.SchemaHelper.class)
@Data
public class Schema implements Serializable {

    private List<String> columnNames;
    @JsonProperty("columns")
    private List<ColumnMetaData> columnMetaData;
    private Map<String, Integer> columnNamesIndex; //For efficient lookup

    private Schema() {
        //No-arg constructor for Jackson
    }

    protected Schema(Builder builder) {
        this.columnMetaData = builder.columnMetaData;
        this.columnNames = new ArrayList<>();
        for (ColumnMetaData meta : this.columnMetaData)
            this.columnNames.add(meta.getName());
        columnNamesIndex = new HashMap<>();
        for (int i = 0; i < columnNames.size(); i++) {
            columnNamesIndex.put(columnNames.get(i), i);
        }
    }

    /**
     * Create a schema based on the
     * given metadata
     * @param columnMetaData the metadata to create the
     *                       schema from
     */
    public Schema(@JsonProperty("columns") List<ColumnMetaData> columnMetaData) {
        if (columnMetaData == null || columnMetaData.size() == 0)
            throw new IllegalArgumentException("Column meta data must be non-empty");
        this.columnMetaData = columnMetaData;
        this.columnNames = new ArrayList<>();
        for (ColumnMetaData meta : this.columnMetaData)
            this.columnNames.add(meta.getName());
        this.columnNamesIndex = new HashMap<>();
        for (int i = 0; i < columnNames.size(); i++) {
            columnNamesIndex.put(columnNames.get(i), i);
        }
    }


    /**
     * Returns true if the given schema
     * has the same types at each index
     * @param schema the schema to compare the types to
     * @return true if the schema has the same types
     * at every index as this one,false otherwise
     */
    public boolean sameTypes(Schema schema) {
        if (schema.numColumns() != numColumns())
            return false;
        for (int i = 0; i < schema.numColumns(); i++) {
            if (getType(i) != schema.getType(i))
                return false;
        }

        return true;
    }

    /**
     * Compute the difference in {@link ColumnMetaData}
     * between this schema and the passed in schema.
     * This is useful during the {@link org.datavec.api.transform.TransformProcess}
     * to identify what a process will do to a given {@link Schema}.
     *
     * @param schema the schema to compute the difference for
     * @return the metadata that is different (in order)
     * between this schema and the other schema
     */
    public List<ColumnMetaData> differences(Schema schema) {
        List<ColumnMetaData> ret = new ArrayList<>();
        for (int i = 0; i < schema.numColumns(); i++) {
            if (!columnMetaData.contains(schema.getMetaData(i)))
                ret.add(schema.getMetaData(i));
        }

        return ret;
    }

    /**
     * Create a new schema based on the new metadata
     * @param columnMetaData the new metadata to create the
     *                       schema from
     * @return the new schema
     */
    public Schema newSchema(List<ColumnMetaData> columnMetaData) {
        return new Schema(columnMetaData);
    }

    /**
     * Returns the number of columns or fields
     * for this schema
     * @return the number of columns or fields for this schema
     */
    public int numColumns() {
        return columnNames.size();
    }

    /**
     * Returns the name of a
     * given column at the specified index
     * @param column the index of the column
     *               to get the name for
     * @return the name of the column at the specified index
     */
    public String getName(int column) {
        return columnNames.get(column);
    }

    /**
     * Returns the {@link ColumnType}
     * for the column at the specified index
     * @param column the index of the column to get the type for
     * @return the type of the column to at the specified inde
     */
    public ColumnType getType(int column) {
        if (column < 0 || column >= columnMetaData.size())
            throw new IllegalArgumentException(
                            "Invalid column number. " + column + "only " + columnMetaData.size() + "present.");
        return columnMetaData.get(column).getColumnType();
    }

    /**
     * Returns the {@link ColumnType}
     * for the column at the specified index
     * @param columnName the index of the column to get the type for
     * @return the type of the column to at the specified inde
     */
    public ColumnType getType(String columnName) {
        if (!hasColumn(columnName)) {
            throw new IllegalArgumentException("Column \"" + columnName + "\" does not exist in schema");
        }
        return getMetaData(columnName).getColumnType();
    }

    /**
     * Returns the {@link ColumnMetaData}
     * at the specified column index
     * @param column the index
     *               to get the metadata for
     * @return the metadata at ths specified index
     */
    public ColumnMetaData getMetaData(int column) {
        return columnMetaData.get(column);
    }

    /**
     * Retrieve the metadata for the given
     * column name
     * @param column the name of the column to get metadata for
     * @return the metadata for the given column name
     */
    public ColumnMetaData getMetaData(String column) {
        return getMetaData(getIndexOfColumn(column));
    }

    /**
     * Return a copy of the list column names
     * @return a copy of the list of column names
     * for this schema
     */
    public List<String> getColumnNames() {
        return new ArrayList<>(columnNames);
    }

    /**
     * A copy of the list of {@link ColumnType}
     * for this schema
     * @return the list of column  types in order based
     * on column index for this schema
     */
    public List<ColumnType> getColumnTypes() {
        List<ColumnType> list = new ArrayList<>(columnMetaData.size());
        for (ColumnMetaData md : columnMetaData)
            list.add(md.getColumnType());
        return list;
    }

    /**
     * Returns a copy of the underlying
     * schema {@link ColumnMetaData}
     * @return the list of schema metadata
     */
    public List<ColumnMetaData> getColumnMetaData() {
        return new ArrayList<>(columnMetaData);
    }

    /**
     * Returns the index for the given
     * column name
     * @param columnName the column name to get the
     *                   index for
     * @return the index of the given column name
     * for the schema
     */
    public int getIndexOfColumn(String columnName) {
        Integer idx = columnNamesIndex.get(columnName);
        if (idx == null)
            throw new NoSuchElementException("Unknown column: \"" + columnName + "\"");
        return idx;
    }

    /**
     * Return the indices of the columns, given their namess
     *
     * @param columnNames Name of the columns to get indices for
     * @return Column indexes
     */
    public int[] getIndexOfColumns(Collection<String> columnNames) {
        return getIndexOfColumns(columnNames.toArray(new String[columnNames.size()]));
    }

    /**
     * Return the indices of the columns, given their namess
     *
     * @param columnNames Name of the columns to get indices for
     * @return Column indexes
     */
    public int[] getIndexOfColumns(String... columnNames) {
        int[] out = new int[columnNames.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = getIndexOfColumn(columnNames[i]);
        }
        return out;
    }

    /**
     * Determine if the schema has a column with the specified name
     *
     * @param columnName Name to see if the column exists
     * @return True if a column exists for that name, false otherwise
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
            sb.append(String.format("%-6d", i)).append(paddedName).append(String.format("%-15s", type)).append(meta)
                            .append("\n");
        }

        return sb.toString();
    }

    /**
     * Serialize this schema to json
     * @return a json representation of this schema
     */
    public String toJson() {
        return toJacksonString(new JsonFactory());
    }

    /**
     * Serialize this schema to yaml
     * @return the yaml representation of this schema
     */
    public String toYaml() {
        return toJacksonString(new YAMLFactory());
    }

    private String toJacksonString(JsonFactory factory) {
        ObjectMapper om = new ObjectMapper(factory);
        om.registerModule(new JodaModule());
        om.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        om.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        om.enable(SerializationFeature.INDENT_OUTPUT);
        om.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.NONE);
        om.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
        String str;
        try {
            str = om.writeValueAsString(this);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return str;
    }

    /**
     * Create a schema from a given json string
     * @param json the json to create the schema from
     * @return the created schema based on the json
     */
    public static Schema fromJson(String json) {
        try{
            return JsonMappers.getMapper().readValue(json, Schema.class);
        } catch (Exception e){
            //TODO better exceptions
            throw new RuntimeException(e);
        }
    }

    /**
     * Create a schema from the given
     * yaml string
     * @param yaml the yaml to create the schema from
     * @return the created schema based on the yaml
     */
    public static Schema fromYaml(String yaml) {
        try{
            return JsonMappers.getMapperYaml().readValue(yaml, Schema.class);
        } catch (Exception e){
            //TODO better exceptions
            throw new RuntimeException(e);
        }
    }

    private static Schema fromJacksonString(String str, JsonFactory factory) {
        ObjectMapper om = new ObjectMapper(factory);
        om.registerModule(new JodaModule());
        om.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        om.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        om.enable(SerializationFeature.INDENT_OUTPUT);
        om.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.NONE);
        om.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
        try {
            return om.readValue(str, Schema.class);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static class Builder {
        List<ColumnMetaData> columnMetaData = new ArrayList<>();

        /**
         * Add a Float column with no restrictions on the allowable values, except for no NaN/infinite values allowed
         *
         * @param name Name of the column
         */
        public Builder addColumnFloat(String name) {
            return addColumn(new FloatMetaData(name));
        }

        /**
         * Add a Double column with no restrictions on the allowable values, except for no NaN/infinite values allowed
         *
         * @param name Name of the column
         */
        public Builder addColumnDouble(String name) {
            return addColumn(new DoubleMetaData(name));
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
         * Add a Double column with the specified restrictions
         *
         * @param name            Name of the column
         * @param minAllowedValue Minimum allowed value (inclusive). If null: no restriction
         * @param maxAllowedValue Maximum allowed value (inclusive). If null: no restriction
         * @param allowNaN        If false: don't allow NaN values. If true: allow.
         * @param allowInfinite   If false: don't allow infinite values. If true: allow
         */
        public Builder addColumnDouble(String name, Double minAllowedValue, Double maxAllowedValue, boolean allowNaN,
                        boolean allowInfinite) {
            return addColumn(new DoubleMetaData(name, minAllowedValue, maxAllowedValue, allowNaN, allowInfinite));
        }

        /**
         * Add multiple Double columns with no restrictions on the allowable values of the columns (other than no NaN/Infinite)
         *
         * @param columnNames Names of the columns to add
         */
        public Builder addColumnsDouble(String... columnNames) {
            for (String s : columnNames)
                addColumnDouble(s);
            return this;
        }

        /**
         * A convenience method for adding multiple Double columns.
         * For example, to add columns "myDoubleCol_0", "myDoubleCol_1", "myDoubleCol_2", use
         * {@code addColumnsDouble("myDoubleCol_%d",0,2)}
         *
         * @param pattern         Pattern to use (via String.format). "%d" is replaced with column numbers
         * @param minIdxInclusive Minimum column index to use (inclusive)
         * @param maxIdxInclusive Maximum column index to use (inclusive)
         */
        public Builder addColumnsDouble(String pattern, int minIdxInclusive, int maxIdxInclusive) {
            return addColumnsDouble(pattern, minIdxInclusive, maxIdxInclusive, null, null, false, false);
        }

        /**
         * A convenience method for adding multiple Double columns, with additional restrictions that apply to all columns
         * For example, to add columns "myDoubleCol_0", "myDoubleCol_1", "myDoubleCol_2", use
         * {@code addColumnsDouble("myDoubleCol_%d",0,2,null,null,false,false)}
         *
         * @param pattern         Pattern to use (via String.format). "%d" is replaced with column numbers
         * @param minIdxInclusive Minimum column index to use (inclusive)
         * @param maxIdxInclusive Maximum column index to use (inclusive)
         * @param minAllowedValue Minimum allowed value (inclusive). If null: no restriction
         * @param maxAllowedValue Maximum allowed value (inclusive). If null: no restriction
         * @param allowNaN        If false: don't allow NaN values. If true: allow.
         * @param allowInfinite   If false: don't allow infinite values. If true: allow
         */
        public Builder addColumnsDouble(String pattern, int minIdxInclusive, int maxIdxInclusive,
                        Double minAllowedValue, Double maxAllowedValue, boolean allowNaN, boolean allowInfinite) {
            for (int i = minIdxInclusive; i <= maxIdxInclusive; i++) {
                addColumnDouble(String.format(pattern, i), minAllowedValue, maxAllowedValue, allowNaN, allowInfinite);
            }
            return this;
        }

        /**
         * Add an Integer column with no restrictions on the allowable values
         *
         * @param name Name of the column
         */
        public Builder addColumnInteger(String name) {
            return addColumn(new IntegerMetaData(name));
        }

        /**
         * Add an Integer column with the specified min/max allowable values
         *
         * @param name            Name of the column
         * @param minAllowedValue Minimum allowed value (inclusive). If null: no restriction
         * @param maxAllowedValue Maximum allowed value (inclusive). If null: no restriction
         */
        public Builder addColumnInteger(String name, Integer minAllowedValue, Integer maxAllowedValue) {
            return addColumn(new IntegerMetaData(name, minAllowedValue, maxAllowedValue));
        }

        /**
         * Add multiple Integer columns with no restrictions on the min/max allowable values
         *
         * @param names Names of the integer columns to add
         */
        public Builder addColumnsInteger(String... names) {
            for (String s : names)
                addColumnInteger(s);
            return this;
        }

        /**
         * A convenience method for adding multiple Integer columns.
         * For example, to add columns "myIntegerCol_0", "myIntegerCol_1", "myIntegerCol_2", use
         * {@code addColumnsInteger("myIntegerCol_%d",0,2)}
         *
         * @param pattern         Pattern to use (via String.format). "%d" is replaced with column numbers
         * @param minIdxInclusive Minimum column index to use (inclusive)
         * @param maxIdxInclusive Maximum column index to use (inclusive)
         */
        public Builder addColumnsInteger(String pattern, int minIdxInclusive, int maxIdxInclusive) {
            return addColumnsInteger(pattern, minIdxInclusive, maxIdxInclusive, null, null);
        }

        /**
         * A convenience method for adding multiple Integer columns.
         * For example, to add columns "myIntegerCol_0", "myIntegerCol_1", "myIntegerCol_2", use
         * {@code addColumnsInteger("myIntegerCol_%d",0,2)}
         *
         * @param pattern         Pattern to use (via String.format). "%d" is replaced with column numbers
         * @param minIdxInclusive Minimum column index to use (inclusive)
         * @param maxIdxInclusive Maximum column index to use (inclusive)
         * @param minAllowedValue Minimum allowed value (inclusive). If null: no restriction
         * @param maxAllowedValue Maximum allowed value (inclusive). If null: no restriction
         */
        public Builder addColumnsInteger(String pattern, int minIdxInclusive, int maxIdxInclusive,
                        Integer minAllowedValue, Integer maxAllowedValue) {
            for (int i = minIdxInclusive; i <= maxIdxInclusive; i++) {
                addColumnInteger(String.format(pattern, i), minAllowedValue, maxAllowedValue);
            }
            return this;
        }

        /**
         * Add a Categorical column, with the specified state names
         *
         * @param name       Name of the column
         * @param stateNames Names of the allowable states for this categorical column
         */
        public Builder addColumnCategorical(String name, String... stateNames) {
            return addColumn(new CategoricalMetaData(name, stateNames));
        }

        /**
         * Add a Categorical column, with the specified state names
         *
         * @param name       Name of the column
         * @param stateNames Names of the allowable states for this categorical column
         */
        public Builder addColumnCategorical(String name, List<String> stateNames) {
            return addColumn(new CategoricalMetaData(name, stateNames));
        }

        /**
         * Add a Long column, with no restrictions on the min/max values
         *
         * @param name Name of the column
         */
        public Builder addColumnLong(String name) {
            return addColumn(new LongMetaData(name));
        }

        /**
         * Add a Long column with the specified min/max allowable values
         *
         * @param name            Name of the column
         * @param minAllowedValue Minimum allowed value (inclusive). If null: no restriction
         * @param maxAllowedValue Maximum allowed value (inclusive). If null: no restriction
         */
        public Builder addColumnLong(String name, Long minAllowedValue, Long maxAllowedValue) {
            return addColumn(new LongMetaData(name, minAllowedValue, maxAllowedValue));
        }

        /**
         * Add multiple Long columns, with no restrictions on the allowable values
         *
         * @param names Names of the Long columns to add
         */
        public Builder addColumnsLong(String... names) {
            for (String s : names)
                addColumnLong(s);
            return this;
        }

        /**
         * A convenience method for adding multiple Long columns.
         * For example, to add columns "myLongCol_0", "myLongCol_1", "myLongCol_2", use
         * {@code addColumnsLong("myLongCol_%d",0,2)}
         *
         * @param pattern         Pattern to use (via String.format). "%d" is replaced with column numbers
         * @param minIdxInclusive Minimum column index to use (inclusive)
         * @param maxIdxInclusive Maximum column index to use (inclusive)
         */
        public Builder addColumnsLong(String pattern, int minIdxInclusive, int maxIdxInclusive) {
            return addColumnsLong(pattern, minIdxInclusive, maxIdxInclusive, null, null);
        }

        /**
         * A convenience method for adding multiple Long columns.
         * For example, to add columns "myLongCol_0", "myLongCol_1", "myLongCol_2", use
         * {@code addColumnsLong("myLongCol_%d",0,2)}
         *
         * @param pattern         Pattern to use (via String.format). "%d" is replaced with column numbers
         * @param minIdxInclusive Minimum column index to use (inclusive)
         * @param maxIdxInclusive Maximum column index to use (inclusive)
         * @param minAllowedValue Minimum allowed value (inclusive). If null: no restriction
         * @param maxAllowedValue Maximum allowed value (inclusive). If null: no restriction
         */
        public Builder addColumnsLong(String pattern, int minIdxInclusive, int maxIdxInclusive, Long minAllowedValue,
                        Long maxAllowedValue) {
            for (int i = minIdxInclusive; i <= maxIdxInclusive; i++) {
                addColumnLong(String.format(pattern, i), minAllowedValue, maxAllowedValue);
            }
            return this;
        }


        /**
         * Add a column
         *
         * @param metaData metadata for this column
         */
        public Builder addColumn(ColumnMetaData metaData) {
            columnMetaData.add(metaData);
            return this;
        }

        /**
         * Add a String column with no restrictions on the allowable values.
         *
         * @param name Name of  the column
         */
        public Builder addColumnString(String name) {
            return addColumn(new StringMetaData(name));
        }

        /**
         * Add multiple String columns with no restrictions on the allowable values
         *
         * @param columnNames Names of the String columns to add
         */
        public Builder addColumnsString(String... columnNames) {
            for (String s : columnNames)
                addColumnString(s);
            return this;
        }

        /**
         * Add a String column with the specified restrictions
         *
         * @param name               Name of the column
         * @param regex              Regex that the String must match in order to be considered valid. If null: no regex restriction
         * @param minAllowableLength Minimum allowable length for the String to be considered valid
         * @param maxAllowableLength Maximum allowable length for the String to be considered valid
         */
        public Builder addColumnString(String name, String regex, Integer minAllowableLength,
                        Integer maxAllowableLength) {
            return addColumn(new StringMetaData(name, regex, minAllowableLength, maxAllowableLength));
        }

        /**
         * A convenience method for adding multiple numbered String columns.
         * For example, to add columns "myStringCol_0", "myStringCol_1", "myStringCol_2", use
         * {@code addColumnsString("myStringCol_%d",0,2)}
         *
         * @param pattern         Pattern to use (via String.format). "%d" is replaced with column numbers
         * @param minIdxInclusive Minimum column index to use (inclusive)
         * @param maxIdxInclusive Maximum column index to use (inclusive)
         */
        public Builder addColumnsString(String pattern, int minIdxInclusive, int maxIdxInclusive) {
            return addColumnsString(pattern, minIdxInclusive, maxIdxInclusive, null, null, null);
        }

        /**
         * A convenience method for adding multiple numbered String columns.
         * For example, to add columns "myStringCol_0", "myStringCol_1", "myStringCol_2", use
         * {@code addColumnsString("myStringCol_%d",0,2)}
         *
         * @param pattern          Pattern to use (via String.format). "%d" is replaced with column numbers
         * @param minIdxInclusive  Minimum column index to use (inclusive)
         * @param maxIdxInclusive  Maximum column index to use (inclusive)
         * @param regex            Regex that the String must match in order to be considered valid. If null: no regex restriction
         * @param minAllowedLength Minimum allowed length of strings (inclusive). If null: no restriction
         * @param maxAllowedLength Maximum allowed length of strings (inclusive). If null: no restriction
         */
        public Builder addColumnsString(String pattern, int minIdxInclusive, int maxIdxInclusive, String regex,
                        Integer minAllowedLength, Integer maxAllowedLength) {
            for (int i = minIdxInclusive; i <= maxIdxInclusive; i++) {
                addColumnString(String.format(pattern, i), regex, minAllowedLength, maxAllowedLength);
            }
            return this;
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
            addColumn(new TimeMetaData(columnName, timeZone, minValidValue, maxValidValue));
            return this;
        }

        /**
         * Add a NDArray column
         *
         * @param columnName Name of the column
         * @param shape      shape of the NDArray column. Use -1 in entries to specify as "variable length" in that dimension
         */
        public Builder addColumnNDArray(String columnName, long[] shape) {
            return addColumn(new NDArrayMetaData(columnName, shape));
        }

        /**
         * Create the Schema
         */
        public Schema build() {
            return new Schema(this);
        }
    }

    /**
     * Infers a schema based on the record.
     * The column names are based on indexing.
     * @param record the record to infer from
     * @return the infered schema
     */
    public static Schema inferMultiple(List<List<Writable>> record) {
        return infer(record.get(0));
    }

    /**
     * Infers a schema based on the record.
     * The column names are based on indexing.
     * @param record the record to infer from
     * @return the infered schema
     */
    public static Schema infer(List<Writable> record) {
        Schema.Builder builder = new Schema.Builder();
        for (int i = 0; i < record.size(); i++) {
            if (record.get(i) instanceof DoubleWritable)
                builder.addColumnDouble(String.valueOf(i));
            else if (record.get(i) instanceof IntWritable)
                builder.addColumnInteger(String.valueOf(i));
            else if (record.get(i) instanceof LongWritable)
                builder.addColumnLong(String.valueOf(i));
            else if (record.get(i) instanceof FloatWritable)
                builder.addColumnFloat(String.valueOf(i));
            else if(record.get(i) instanceof Text) {
                builder.addColumnString(String.valueOf(i));
            }

            else
                throw new IllegalStateException("Illegal writable for infering schema of type "
                                + record.get(i).getClass().toString() + " with record " + record);
        }

        return builder.build();
    }

}
