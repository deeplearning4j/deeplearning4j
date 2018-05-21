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

package org.datavec.api.transform.stringreduce;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.StringReduceOp;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.StringMetaData;
import org.datavec.api.transform.reduce.ColumnReduction;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.*;

/**
 * A StringReducer is used to take a set of examples and reduce them.
 * The idea: suppose you have a large number of columns, and you want to combine/reduce the values in each column.<br>
 * StringReducer allows you to specify different reductions for differently for different columns: min, max, sum, mean etc.
 * See {@link Builder} and {@link ReduceOp} for the full list.<br>
 * <p>
 * Uses are:
 * (1) Reducing examples by a key
 * (2) Reduction operations in time series (windowing ops, etc)
 *
 * @author Alex Black
 */
@Data
@JsonIgnoreProperties({"schema", "keyColumnsSet"})
@EqualsAndHashCode(exclude = {"schema", "keyColumnsSet"})
public class StringReducer implements IStringReducer {

    private Schema schema;
    private final List<String> inputColumns;
    private final Set<String> inputColumnsSet;
    private String outputColumnName;
    private final StringReduceOp stringReduceOp;
    private Map<String, ColumnReduction> customReductions;

    private StringReducer(Builder builder) {
        this(builder.inputColumns, builder.defaultOp, builder.customReductions, builder.outputColumnName);
    }

    public StringReducer(@JsonProperty("inputColumns") List<String> inputColumns,
                    @JsonProperty("op") StringReduceOp stringReduceOp,
                    @JsonProperty("customReductions") Map<String, ColumnReduction> customReductions,
                    @JsonProperty("outputColumnName") String outputColumnName) {
        this.inputColumns = inputColumns;
        this.inputColumnsSet = (inputColumns == null ? null : new HashSet<>(inputColumns));
        this.stringReduceOp = stringReduceOp;
        this.customReductions = customReductions;
        this.outputColumnName = outputColumnName;
    }

    @Override
    public void setInputSchema(Schema schema) {
        this.schema = schema;
    }

    @Override
    public Schema getInputSchema() {
        return schema;
    }

    @Override
    public List<String> getInputColumns() {
        return inputColumns;
    }

    /**
     * Get the output schema, given the input schema
     */
    @Override
    public Schema transform(Schema schema) {
        int nCols = schema.numColumns();
        List<ColumnMetaData> meta = schema.getColumnMetaData();
        List<ColumnMetaData> newMeta = new ArrayList<>(nCols);
        newMeta.addAll(meta);
        newMeta.add(new StringMetaData(outputColumnName));
        return schema.newSchema(newMeta);
    }

    private static ColumnMetaData getMetaForColumn(StringReduceOp op, String name, ColumnMetaData inMeta) {
        inMeta = inMeta.clone();
        switch (op) {
            case PREPEND:
                inMeta.setName("prepend(" + name + ")");
                return inMeta;
            case APPEND:
                inMeta.setName("append(" + name + ")");
                return inMeta;
            case REPLACE:
                inMeta.setName("replace(" + name + ")");
                return inMeta;
            case MERGE:
                inMeta.setName("merge(" + name + ")");
                return inMeta;
            default:
                throw new UnsupportedOperationException("Unknown or not implemented op: " + op);
        }
    }

    @Override
    public List<Writable> reduce(List<List<Writable>> examplesList) {
        //Go through each writable, and reduce according to whatever strategy is specified

        if (schema == null)
            throw new IllegalStateException("Error: Schema has not been set");


        List<Writable> out = new ArrayList<>(examplesList.size());
        for (int i = 0; i < examplesList.size(); i++) {
            out.add(reduceStringOrCategoricalColumn(stringReduceOp, examplesList.get(i)));
        }

        return out;
    }



    public static Writable reduceStringOrCategoricalColumn(StringReduceOp op, List<Writable> values) {
        switch (op) {
            case MERGE:
            case APPEND:
                StringBuilder stringBuilder = new StringBuilder();
                for (Writable w : values) {
                    stringBuilder.append(w.toString());
                }
                return new Text(stringBuilder.toString());
            case REPLACE:
                if (values.size() > 2) {
                    throw new IllegalArgumentException("Unable to run replace on columns > 2");
                }
                return new Text(values.get(1).toString());
            case PREPEND:
                List<Writable> reverse = new ArrayList<>(values);
                Collections.reverse(reverse);
                StringBuilder stringBuilder2 = new StringBuilder();
                for (Writable w : reverse) {
                    stringBuilder2.append(w.toString());
                }

                return new Text(stringBuilder2.toString());
            default:
                throw new UnsupportedOperationException("Cannot execute op \"" + op + "\" on String/Categorical column "
                                + "(can only perform Count, CountUnique, TakeFirst and TakeLast ops on categorical columns)");
        }
    }



    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("StringReducer(");

        sb.append("defaultOp=").append(stringReduceOp);

        if (customReductions != null) {
            sb.append(",customReductions=").append(customReductions);
        }


        sb.append(")");
        return sb.toString();
    }


    public static class Builder {

        private StringReduceOp defaultOp;
        private Map<String, StringReduceOp> opMap = new HashMap<>();
        private Map<String, ColumnReduction> customReductions = new HashMap<>();
        private Set<String> ignoreInvalidInColumns = new HashSet<>();
        private String outputColumnName;
        private List<String> inputColumns;



        public Builder inputColumns(List<String> inputColumns) {
            this.inputColumns = inputColumns;
            return this;
        }

        /**
         * Create a StringReducer builder, and set the default column reduction operation.
         * For any columns that aren't specified explicitly, they will use the default reduction operation.
         * If a column does have a reduction operation explicitly specified, then it will override
         * the default specified here.
         *
         * @param defaultOp Default reduction operation to perform
         */
        public Builder(StringReduceOp defaultOp) {
            this.defaultOp = defaultOp;
        }

        public Builder outputColumnName(String outputColumnName) {
            this.outputColumnName = outputColumnName;
            return this;
        }


        private Builder add(StringReduceOp op, String[] cols) {
            for (String s : cols) {
                opMap.put(s, op);
            }
            return this;
        }

        /**
         * Reduce the specified columns by taking the minimum value
         */
        public Builder appendColumns(String... columns) {
            return add(StringReduceOp.APPEND, columns);
        }

        /**
         * Reduce the specified columns by taking the maximum value
         */
        public Builder prependColumns(String... columns) {
            return add(StringReduceOp.PREPEND, columns);
        }

        /**
         * Reduce the specified columns by taking the sum of values
         */
        public Builder mergeColumns(String... columns) {
            return add(StringReduceOp.MERGE, columns);
        }

        /**
         * Reduce the specified columns by taking the mean of the values
         */
        public Builder replaceColumn(String... columns) {
            return add(StringReduceOp.REPLACE, columns);
        }

        /**
         * Reduce the specified column using a custom column reduction functionality.
         *
         * @param column          Column to execute the custom reduction functionality on
         * @param columnReduction Column reduction to execute on that column
         */
        public Builder customReduction(String column, ColumnReduction columnReduction) {
            customReductions.put(column, columnReduction);
            return this;
        }

        /**
         * When doing the reduction: set the specified columns to ignore any invalid values.
         * Invalid: defined as being not valid according to the ColumnMetaData: {@link ColumnMetaData#isValid(Writable)}.
         * For numerical columns, this typically means being unable to parse the Writable. For example, Writable.toLong() failing for a Long column.
         * If the column has any restrictions (min/max values, regex for Strings etc) these will also be taken into account.
         *
         * @param columns Columns to set 'ignore invalid' for
         */
        public Builder setIgnoreInvalid(String... columns) {
            Collections.addAll(ignoreInvalidInColumns, columns);
            return this;
        }

        public StringReducer build() {
            return new StringReducer(this);
        }
    }


}
