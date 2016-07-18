/*
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

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.dataformat.xml.XmlFactory;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.metadata.*;
import org.joda.time.DateTimeZone;

import java.io.Serializable;
import java.util.*;

/**
 * Created by Alex on 4/03/2016.
 */
@JsonIgnoreProperties({"columnNames","columnNamesIndex"})
public class Schema implements Serializable {

    private List<String> columnNames;
    private List<ColumnMetaData> columnMetaData;
    private Map<String, Integer> columnNamesIndex;   //For efficient lookup


    protected Schema(Builder builder) {
        this.columnMetaData = builder.columnMetaData;
        this.columnNames = new ArrayList<>();
        for(ColumnMetaData meta : this.columnMetaData) this.columnNames.add(meta.getColumnName());
        columnNamesIndex = new HashMap<>();
        for (int i = 0; i < columnNames.size(); i++) {
            columnNamesIndex.put(columnNames.get(i), i);
        }
    }

    public Schema(List<ColumnMetaData> columnMetaData) {
        if (columnMetaData == null || columnMetaData.size() == 0) throw new IllegalArgumentException("Column meta data must be non-empty");
        this.columnMetaData = columnMetaData;
        this.columnNames = new ArrayList<>();
        for(ColumnMetaData meta : this.columnMetaData) this.columnNames.add(meta.getColumnName());
        this.columnNamesIndex = new HashMap<>();
        for (int i = 0; i < columnNames.size(); i++) {
            columnNamesIndex.put(columnNames.get(i), i);
        }
    }


    public Schema newSchema(List<ColumnMetaData> columnMetaData) {
        return new Schema(columnMetaData);
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

    public String toJson(){
        return toJacksonString(new JsonFactory());
    }

    public String toXml(){
        return toJacksonString(new XmlFactory());
    }

    public String toYaml(){
        return toJacksonString(new YAMLFactory());
    }

    private String toJacksonString(JsonFactory factory){
        ObjectMapper om = new ObjectMapper(factory);
        om.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        om.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        om.enable(SerializationFeature.INDENT_OUTPUT);
        String str;
        try{
            str = om.writeValueAsString(this);
        } catch(Exception e){
            throw new RuntimeException(e);
        }

        return str;
    }

    public static Schema fromJson(String json){
        return fromJacksonString(json, new JsonFactory());
    }

    public static Schema fromXml(String xml){
        return fromJacksonString(xml, new XmlFactory());
    }

    public static Schema fromYaml(String yaml){
        return fromJacksonString(yaml, new YAMLFactory());
    }

    private static Schema fromJacksonString(String str, JsonFactory factory){
        ObjectMapper om = new ObjectMapper(factory);
        try{
            return om.readValue(str, Schema.class);
        }catch(Exception e){
            throw new RuntimeException(e);
        }
    }

    public static class Builder {

//        List<String> columnNames = new ArrayList<>();
        List<ColumnMetaData> columnMetaData = new ArrayList<>();

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
            return addColumn(new DoubleMetaData(name, minAllowedValue, maxAllowedValue, allowNaN, allowInfinite));
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
            return addColumn(new IntegerMetaData(name));
        }

        /**
         * Add an integer column with the specified min/max allowable values
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
         * Add an Long column with the specified min/max allowable values
         *
         * @param name            Name of the column
         * @param minAllowedValue Minimum allowed value (inclusive). If null: no restriction
         * @param maxAllowedValue Maximum allowed value (inclusive). If null: no restriction
         */
        public Builder addColumnLong(String name, Long minAllowedValue, Long maxAllowedValue) {
            return addColumn(new LongMetaData(name, minAllowedValue, maxAllowedValue));
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
         * @param metaData metadata for this column
         */
//        public Builder addColumn(String name, ColumnMetaData metaData) {
        public Builder addColumn(ColumnMetaData metaData) {
//            columnNames.add(name);
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
            return addColumn(new StringMetaData(name, regex, minAllowableLength, maxAllowableLength));
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
         * Create the Schema
         */
        public Schema build() {
            return new Schema(this);
        }
    }

}
