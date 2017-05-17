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

package org.datavec.api.transform.reduce;

import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import lombok.EqualsAndHashCode;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.DoubleMetaData;
import org.datavec.api.transform.metadata.IntegerMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.metadata.LongMetaData;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;

import java.io.Serializable;
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
public class Reducer implements IReducer {

    private Schema schema;
    private final List<String> keyColumns;
    private final Set<String> keyColumnsSet;
    private final ReduceOp defaultOp;
    private final Map<String, ReduceOp> opMap;
    private Map<String, ColumnReduction> customReductions;
    private Map<String, ConditionalReduction> conditionalReductions;
    private Set<String> ignoreInvalidInColumns;

    private Reducer(Builder builder) {
        this((builder.keyColumns == null ? null : Arrays.asList(builder.keyColumns)), builder.defaultOp, builder.opMap,
                        builder.customReductions, builder.conditionalReductions, builder.ignoreInvalidInColumns);
    }

    public Reducer(@JsonProperty("keyColumns") List<String> keyColumns, @JsonProperty("defaultOp") ReduceOp defaultOp,
                    @JsonProperty("opMap") Map<String, ReduceOp> opMap,
                    @JsonProperty("customReductions") Map<String, ColumnReduction> customReductions,
                    @JsonProperty("conditionalReductions") Map<String, ConditionalReduction> conditionalReductions,
                    @JsonProperty("ignoreInvalidInColumns") Set<String> ignoreInvalidInColumns) {
        this.keyColumns = keyColumns;
        this.keyColumnsSet = (keyColumns == null ? null : new HashSet<>(keyColumns));
        this.defaultOp = defaultOp;
        this.opMap = opMap;
        this.customReductions = customReductions;
        this.conditionalReductions = conditionalReductions;
        this.ignoreInvalidInColumns = ignoreInvalidInColumns;
    }

    @Override
    public void setInputSchema(Schema schema) {
        this.schema = schema;
        //Conditions (if any) also need the input schema:
        for (ConditionalReduction cr : conditionalReductions.values()) {
            cr.getCondition().setInputSchema(schema);
        }
    }

    @Override
    public Schema getInputSchema() {
        return schema;
    }

    @Override
    public List<String> getKeyColumns() {
        return keyColumns;
    }

    /**
     * Get the output schema, given the input schema
     */
    @Override
    public Schema transform(Schema schema) {
        int nCols = schema.numColumns();
        List<String> colNames = schema.getColumnNames();
        List<ColumnMetaData> meta = schema.getColumnMetaData();
        List<ColumnMetaData> newMeta = new ArrayList<>(nCols);

        for (int i = 0; i < nCols; i++) {
            String name = colNames.get(i);
            ColumnMetaData inMeta = meta.get(i);

            if (keyColumnsSet != null && keyColumnsSet.contains(name)) {
                //No change to key columns
                newMeta.add(inMeta);
                continue;
            }

            //First: check for a custom reductions on this column
            if (customReductions != null && customReductions.containsKey(name)) {
                ColumnReduction reduction = customReductions.get(name);

                String outName = reduction.getColumnOutputName(name);
                ColumnMetaData outMeta = reduction.getColumnOutputMetaData(outName, inMeta);

                newMeta.add(outMeta);

                continue;
            }

            //Second: check for conditional reductions on this column:
            if (conditionalReductions != null && conditionalReductions.containsKey(name)) {
                ConditionalReduction reduction = conditionalReductions.get(name);

                String outName = reduction.getOutputName();
                ColumnMetaData m = getMetaForColumn(reduction.getReduction(), name, inMeta);
                m.setName(outName);
                newMeta.add(m);

                continue;
            }

            //Otherwise: get the specified (built-in) reduction op
            //If no reduction op is specified for that column: use the default
            ReduceOp op = opMap.get(name);
            if (op == null)
                op = defaultOp;
            newMeta.add(getMetaForColumn(op, name, inMeta));
        }

        return schema.newSchema(newMeta);
    }

    private static ColumnMetaData getMetaForColumn(ReduceOp op, String name, ColumnMetaData inMeta) {
        inMeta = inMeta.clone();
        switch (op) {
            case Min:
                inMeta.setName("min(" + name + ")");
                return inMeta;
            case Max:
                inMeta.setName("max(" + name + ")");
                return inMeta;
            case Range:
                inMeta.setName("range(" + name + ")");
                return inMeta;
            case TakeFirst:
                inMeta.setName("first(" + name + ")");
                return inMeta;
            case TakeLast:
                inMeta.setName("last(" + name + ")");
                return inMeta;
            case Sum:
                String outName = "sum(" + name + ")";
                //Issue with sum: the input meta data restrictions probably won't hold. But the data _type_ should essentially remain the same
                ColumnMetaData outMeta;
                if (inMeta instanceof IntegerMetaData || inMeta instanceof LongMetaData) {
                    outMeta = new LongMetaData(outName);
                } else if (inMeta instanceof DoubleMetaData) {
                    outMeta = new DoubleMetaData(outName);
                } else {
                    //Sum doesn't really make sense to sum other column types anyway...
                    outMeta = inMeta;
                }
                outMeta.setName(outName);
                return outMeta;
            case Mean:
                return new DoubleMetaData("mean(" + name + ")");
            case Stdev:
                return new DoubleMetaData("stdev(" + name + ")");
            case Count:
                return new IntegerMetaData("count", 0, null);
            case CountUnique:
                //Always integer
                return new IntegerMetaData("countUnique(" + name + ")", 0, null);
            default:
                throw new UnsupportedOperationException("Unknown or not implemented op: " + op);
        }
    }

    @Override
    public List<Writable> reduce(List<List<Writable>> examplesList) {
        //Go through each writable, and reduce according to whatever strategy is specified

        if (schema == null)
            throw new IllegalStateException("Error: Schema has not been set");

        int nCols = schema.numColumns();
        List<String> colNames = schema.getColumnNames();

        List<Writable> out = new ArrayList<>(nCols);
        List<Writable> tempColumnValues = new ArrayList<>(examplesList.size());
        for (int i = 0; i < nCols; i++) {
            String colName = colNames.get(i);
            if (keyColumnsSet != null && keyColumnsSet.contains(colName)) {
                //This is a key column -> all values should be identical
                //Therefore just take the first one
                out.add(examplesList.get(0).get(i));
                continue;
            }

            //First: Extract out the Writables for the column we are considering here...
            for (List<Writable> list : examplesList) {
                tempColumnValues.add(list.get(i));
            }

            //Second: is this a *custom* reduction column?
            if (customReductions != null && customReductions.containsKey(colName)) {
                ColumnReduction reduction = customReductions.get(colName);
                Writable reducedColumn = reduction.reduceColumn(tempColumnValues);
                out.add(reducedColumn);
                tempColumnValues.clear();
                continue;
            }

            //Third: is this a *conditional* reduction column?
            //Only practical difference with conditional reductions is we filter the input based on a condition first
            boolean conditionalOp = false;
            if (conditionalReductions != null && conditionalReductions.containsKey(colName)) {
                ConditionalReduction reduction = conditionalReductions.get(colName);
                Condition c = reduction.getCondition();
                List<Writable> filteredColumnValues = new ArrayList<>();

                int j = 0;
                for (List<Writable> example : examplesList) {
                    if (c.condition(example)) {
                        filteredColumnValues.add(tempColumnValues.get(j));
                    }
                    j++;
                }

                tempColumnValues = filteredColumnValues;
                conditionalOp = true;
            }

            //What type of column is this?
            ColumnType type = schema.getType(i);

            //What op are we performing on this column?
            ReduceOp op = (conditionalOp ? conditionalReductions.get(colName).getReduction() : opMap.get(colName));
            if (op == null)
                op = defaultOp;

            //Execute the reduction, store the result
            out.add(ReductionUtils.reduceColumn(op, type, tempColumnValues, ignoreInvalidInColumns.contains(colName),
                            schema.getMetaData(i)));

            tempColumnValues.clear();
        }

        return out;
    }



    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("StringReducer(");
        if (keyColumns != null) {
            sb.append("keyColumns=").append(keyColumns).append(",");
        }
        sb.append("defaultOp=").append(defaultOp);
        if (opMap != null) {
            sb.append(",opMap=").append(opMap);
        }
        if (customReductions != null) {
            sb.append(",customReductions=").append(customReductions);
        }
        if (conditionalReductions != null) {
            sb.append(",conditionalReductions=").append(conditionalReductions);
        }
        if (ignoreInvalidInColumns != null) {
            sb.append(",ignoreInvalidInColumns=").append(ignoreInvalidInColumns);
        }
        sb.append(")");
        return sb.toString();
    }


    public static class Builder {

        private ReduceOp defaultOp;
        private Map<String, ReduceOp> opMap = new HashMap<>();
        private Map<String, ColumnReduction> customReductions = new HashMap<>();
        private Map<String, ConditionalReduction> conditionalReductions = new HashMap<>();
        private Set<String> ignoreInvalidInColumns = new HashSet<>();
        private String[] keyColumns;


        /**
         * Create a StringReducer builder, and set the default column reduction operation.
         * For any columns that aren't specified explicitly, they will use the default reduction operation.
         * If a column does have a reduction operation explicitly specified, then it will override
         * the default specified here.
         *
         * @param defaultOp Default reduction operation to perform
         */
        public Builder(ReduceOp defaultOp) {
            this.defaultOp = defaultOp;
        }

        /**
         * Specify the key columns. The idea here is to be able to create a (potentially compound) key
         * out of multiple columns, using the toString representation of the values in these columns
         *
         * @param keyColumns Columns that will make up the key
         * @return
         */
        public Builder keyColumns(String... keyColumns) {
            this.keyColumns = keyColumns;
            return this;
        }

        private Builder add(ReduceOp op, String[] cols) {
            for (String s : cols) {
                opMap.put(s, op);
            }
            return this;
        }

        /**
         * Reduce the specified columns by taking the minimum value
         */
        public Builder minColumns(String... columns) {
            return add(ReduceOp.Min, columns);
        }

        /**
         * Reduce the specified columns by taking the maximum value
         */
        public Builder maxColumn(String... columns) {
            return add(ReduceOp.Max, columns);
        }

        /**
         * Reduce the specified columns by taking the sum of values
         */
        public Builder sumColumns(String... columns) {
            return add(ReduceOp.Sum, columns);
        }

        /**
         * Reduce the specified columns by taking the mean of the values
         */
        public Builder meanColumns(String... columns) {
            return add(ReduceOp.Mean, columns);
        }

        /**
         * Reduce the specified columns by taking the standard deviation of the values
         */
        public Builder stdevColumns(String... columns) {
            return add(ReduceOp.Stdev, columns);
        }

        /**
         * Reduce the specified columns by counting the number of values
         */
        public Builder countColumns(String... columns) {
            return add(ReduceOp.Count, columns);
        }

        /**
         * Reduce the specified columns by taking the range (max-min) of the values
         */
        public Builder rangeColumns(String... columns) {
            return add(ReduceOp.Range, columns);
        }

        /**
         * Reduce the specified columns by counting the number of unique values
         */
        public Builder countUniqueColumns(String... columns) {
            return add(ReduceOp.CountUnique, columns);
        }

        /**
         * Reduce the specified columns by taking the first value
         */
        public Builder takeFirstColumns(String... columns) {
            return add(ReduceOp.TakeFirst, columns);
        }

        /**
         * Reduce the specified columns by taking the last value
         */
        public Builder takeLastColumns(String... columns) {
            return add(ReduceOp.TakeLast, columns);
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
         * Conditional reduction: apply the reduce on a specified column, where the reduction occurs *only* on those
         * examples where the condition returns true. Examples where the condition does not apply (returns false) are
         * ignored/excluded.
         *
         * @param column     Name of the column to execute the conditional reduction on
         * @param outputName Name of the column, after the reduction has been executed
         * @param reduction  Reduction to execute
         * @param condition  Condition to use in the reductions
         */
        public Builder conditionalReduction(String column, String outputName, ReduceOp reduction, Condition condition) {
            this.conditionalReductions.put(column, new ConditionalReduction(column, outputName, reduction, condition));
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

        public Reducer build() {
            return new Reducer(this);
        }
    }

    @AllArgsConstructor
    @Data
    public static class ConditionalReduction implements Serializable {
        private final String columnName;
        private final String outputName;
        private final ReduceOp reduction;
        private final Condition condition;
    }

}
