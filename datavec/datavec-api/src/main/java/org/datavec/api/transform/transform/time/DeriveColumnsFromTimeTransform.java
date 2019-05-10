/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.api.transform.transform.time;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.IntegerMetaData;
import org.datavec.api.transform.metadata.StringMetaData;
import org.datavec.api.transform.metadata.TimeMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.jackson.DateTimeFieldTypeDeserializer;
import org.datavec.api.util.jackson.DateTimeFieldTypeSerializer;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.joda.time.DateTime;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Create a number of new columns by deriving their values from a Time column.
 * Can be used for example to create new columns with the year, month, day, hour, minute, second etc.
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"inputSchema", "insertAfterIdx", "deriveFromIdx"})
@EqualsAndHashCode(exclude = {"inputSchema", "insertAfterIdx", "deriveFromIdx"})
@Data
public class DeriveColumnsFromTimeTransform implements Transform {

    private final String columnName;
    private final String insertAfter;
    private DateTimeZone inputTimeZone;
    private final List<DerivedColumn> derivedColumns;
    private Schema inputSchema;
    private int insertAfterIdx = -1;
    private int deriveFromIdx = -1;


    private DeriveColumnsFromTimeTransform(Builder builder) {
        this.derivedColumns = builder.derivedColumns;
        this.columnName = builder.columnName;
        this.insertAfter = builder.insertAfter;
    }

    public DeriveColumnsFromTimeTransform(@JsonProperty("columnName") String columnName,
                    @JsonProperty("insertAfter") String insertAfter,
                    @JsonProperty("inputTimeZone") DateTimeZone inputTimeZone,
                    @JsonProperty("derivedColumns") List<DerivedColumn> derivedColumns) {
        this.columnName = columnName;
        this.insertAfter = insertAfter;
        this.inputTimeZone = inputTimeZone;
        this.derivedColumns = derivedColumns;
    }

    @Override
    public Schema transform(Schema inputSchema) {
        List<ColumnMetaData> oldMeta = inputSchema.getColumnMetaData();
        List<ColumnMetaData> newMeta = new ArrayList<>(oldMeta.size() + derivedColumns.size());

        List<String> oldNames = inputSchema.getColumnNames();

        for (int i = 0; i < oldMeta.size(); i++) {
            String current = oldNames.get(i);
            newMeta.add(oldMeta.get(i));

            if (insertAfter.equals(current)) {
                //Insert the derived columns here
                for (DerivedColumn d : derivedColumns) {
                    switch (d.columnType) {
                        case String:
                            newMeta.add(new StringMetaData(d.columnName));
                            break;
                        case Integer:
                            newMeta.add(new IntegerMetaData(d.columnName)); //TODO: ranges... if it's a day, we know it must be 1 to 31, etc...
                            break;
                        default:
                            throw new IllegalStateException("Unexpected column type: " + d.columnType);
                    }
                }
            }
        }

        return inputSchema.newSchema(newMeta);
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        insertAfterIdx = inputSchema.getColumnNames().indexOf(insertAfter);
        if (insertAfterIdx == -1) {
            throw new IllegalStateException(
                            "Invalid schema/insert after column: input schema does not contain column \"" + insertAfter
                                            + "\"");
        }

        deriveFromIdx = inputSchema.getColumnNames().indexOf(columnName);
        if (deriveFromIdx == -1) {
            throw new IllegalStateException(
                            "Invalid source column: input schema does not contain column \"" + columnName + "\"");
        }

        this.inputSchema = inputSchema;

        if (!(inputSchema.getMetaData(columnName) instanceof TimeMetaData))
            throw new IllegalStateException("Invalid state: input column \"" + columnName
                            + "\" is not a time column. Is: " + inputSchema.getMetaData(columnName));
        TimeMetaData meta = (TimeMetaData) inputSchema.getMetaData(columnName);
        inputTimeZone = meta.getTimeZone();
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        if (writables.size() != inputSchema.numColumns()) {
            throw new IllegalStateException("Cannot execute transform: input writables list length (" + writables.size()
                            + ") does not " + "match expected number of elements (schema: " + inputSchema.numColumns()
                            + "). Transform = " + toString());
        }

        int i = 0;
        Writable source = writables.get(deriveFromIdx);
        List<Writable> list = new ArrayList<>(writables.size() + derivedColumns.size());
        for (Writable w : writables) {
            list.add(w);
            if (i++ == insertAfterIdx) {
                for (DerivedColumn d : derivedColumns) {
                    switch (d.columnType) {
                        case String:
                            list.add(new Text(d.dateTimeFormatter.print(source.toLong())));
                            break;
                        case Integer:
                            DateTime dt = new DateTime(source.toLong(), inputTimeZone);
                            list.add(new IntWritable(dt.get(d.fieldType)));
                            break;
                        default:
                            throw new IllegalStateException("Unexpected column type: " + d.columnType);
                    }
                }
            }
        }
        return list;
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        List<List<Writable>> out = new ArrayList<>(sequence.size());
        for (List<Writable> step : sequence) {
            out.add(map(step));
        }
        return out;
    }

    /**
     * Transform an object
     * in to another object
     *
     * @param input the record to transform
     * @return the transformed writable
     */
    @Override
    public Object map(Object input) {
        List<Object> ret = new ArrayList<>();
        Long l = (Long) input;
        for (DerivedColumn d : derivedColumns) {
            switch (d.columnType) {
                case String:
                    ret.add(d.dateTimeFormatter.print(l));
                    break;
                case Integer:
                    DateTime dt = new DateTime(l, inputTimeZone);
                    ret.add(dt.get(d.fieldType));
                    break;
                default:
                    throw new IllegalStateException("Unexpected column type: " + d.columnType);
            }
        }

        return ret;
    }

    /**
     * Transform a sequence
     *
     * @param sequence
     */
    @Override
    public Object mapSequence(Object sequence) {
        List<Long> longs = (List<Long>) sequence;
        List<List<Object>> ret = new ArrayList<>();
        for (Long l : longs)
            ret.add((List<Object>) map(l));
        return ret;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("DeriveColumnsFromTimeTransform(timeColumn=\"").append(columnName).append("\",insertAfter=\"")
                        .append(insertAfter).append("\",derivedColumns=(");

        boolean first = true;
        for (DerivedColumn d : derivedColumns) {
            if (!first)
                sb.append(",");
            sb.append(d);
            first = false;
        }

        sb.append("))");

        return sb.toString();
    }

    /**
     * The output column name
     * after the operation has been applied
     *
     * @return the output column name
     */
    @Override
    public String outputColumnName() {
        return outputColumnNames()[0];
    }

    /**
     * The output column names
     * This will often be the same as the input
     *
     * @return the output column names
     */
    @Override
    public String[] outputColumnNames() {
        String[] ret = new String[derivedColumns.size()];
        for (int i = 0; i < ret.length; i++)
            ret[i] = derivedColumns.get(i).columnName;
        return ret;
    }

    /**
     * Returns column names
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String[] columnNames() {
        return new String[] {columnName()};
    }

    /**
     * Returns a singular column name
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String columnName() {
        return columnName;
    }

    public static class Builder {

        private final String columnName;
        private String insertAfter;
        private final List<DerivedColumn> derivedColumns = new ArrayList<>();


        /**
         * @param timeColumnName The name of the time column from which to derive the new values
         */
        public Builder(String timeColumnName) {
            this.columnName = timeColumnName;
            this.insertAfter = timeColumnName;
        }

        /**
         * Where should the new columns be inserted?
         * By default, they will be inserted after the source column
         *
         * @param columnName Name of the column to insert the derived columns after
         */
        public Builder insertAfter(String columnName) {
            this.insertAfter = columnName;
            return this;
        }

        /**
         * Add a String column (for example, human readable format), derived from the time
         *
         * @param columnName Name of the new/derived column
         * @param format     Joda time format, as per <a href="http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html">http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html</a>
         * @param timeZone   Timezone to use for formatting
         */
        public Builder addStringDerivedColumn(String columnName, String format, DateTimeZone timeZone) {
            derivedColumns.add(new DerivedColumn(columnName, ColumnType.String, format, timeZone, null));
            return this;
        }

        /**
         * Add an integer derived column - for example, the hour of day, etc. Uses timezone from the time column metadata
         *
         * @param columnName Name of the column
         * @param type       Type of field (for example, DateTimeFieldType.hourOfDay() etc)
         */
        public Builder addIntegerDerivedColumn(String columnName, DateTimeFieldType type) {
            derivedColumns.add(new DerivedColumn(columnName, ColumnType.Integer, null, null, type));
            return this;
        }

        /**
         * Create the transform instance
         */
        public DeriveColumnsFromTimeTransform build() {
            return new DeriveColumnsFromTimeTransform(this);
        }
    }

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @EqualsAndHashCode(exclude = "dateTimeFormatter")
    @Data
    @JsonIgnoreProperties({"dateTimeFormatter"})
    public static class DerivedColumn implements Serializable {
        private final String columnName;
        private final ColumnType columnType;
        private final String format;
        private final DateTimeZone dateTimeZone;
        @JsonSerialize(using = DateTimeFieldTypeSerializer.class)
        @JsonDeserialize(using = DateTimeFieldTypeDeserializer.class)
        private final DateTimeFieldType fieldType;
        private transient DateTimeFormatter dateTimeFormatter;

        //        public DerivedColumn(String columnName, ColumnType columnType, String format, DateTimeZone dateTimeZone, DateTimeFieldType fieldType) {
        public DerivedColumn(@JsonProperty("columnName") String columnName,
                        @JsonProperty("columnType") ColumnType columnType, @JsonProperty("format") String format,
                        @JsonProperty("dateTimeZone") DateTimeZone dateTimeZone,
                        @JsonProperty("fieldType") DateTimeFieldType fieldType) {
            this.columnName = columnName;
            this.columnType = columnType;
            this.format = format;
            this.dateTimeZone = dateTimeZone;
            this.fieldType = fieldType;
            if (format != null)
                dateTimeFormatter = DateTimeFormat.forPattern(this.format).withZone(dateTimeZone);
        }

        @Override
        public String toString() {
            return "(name=" + columnName + ",type=" + columnType + ",derived=" + (format != null ? format : fieldType)
                            + ")";
        }

        //Custom serialization methods, because Joda Time doesn't allow DateTimeFormatter objects to be serialized :(
        private void writeObject(ObjectOutputStream out) throws IOException {
            out.defaultWriteObject();
        }

        private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
            in.defaultReadObject();
            if (format != null)
                dateTimeFormatter = DateTimeFormat.forPattern(format).withZone(dateTimeZone);
        }
    }
}
