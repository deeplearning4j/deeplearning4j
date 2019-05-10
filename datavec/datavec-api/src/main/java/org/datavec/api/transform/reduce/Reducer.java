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

package org.datavec.api.transform.reduce;

import com.clearspring.analytics.util.Preconditions;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.condition.column.TrivialColumnCondition;
import org.datavec.api.transform.metadata.*;
import org.datavec.api.transform.ops.*;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.io.Serializable;
import java.util.*;

/**
 * A Reducer is used to take a set of examples and reduce them.
 * The idea: suppose you have a large number of columns, and you want to combine/reduce the values in each column.<br>
 * Reducer allows you to specify different reductions for differently for different columns: min, max, sum, mean etc.
 * See {@link Builder} and {@link ReduceOp} for the full list.<br>
 * Note this supports executing multipe reducitons per column: simply call the Builder with Xcolumn() repeatedly
 * on the same column, or use {@link Reducer.Builder#multipleOpColmumns(List, String...)}}
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
public class Reducer implements IAssociativeReducer {

    private Schema schema;
    private final List<String> keyColumns;
    private final Set<String> keyColumnsSet;
    private final ReduceOp defaultOp;
    private final Map<String, List<ReduceOp>> opMap;
    private Map<String, ConditionalReduction> conditionalReductions;
    private Map<String, AggregableColumnReduction> customReductions;

    private Set<String> ignoreInvalidInColumns;

    private Reducer(Builder builder) {
        this((builder.keyColumns == null ? null : Arrays.asList(builder.keyColumns)), builder.defaultOp, builder.opMap,
                        builder.customReductions, builder.conditionalReductions, builder.ignoreInvalidInColumns);
    }

    public Reducer(@JsonProperty("keyColumns") List<String> keyColumns, @JsonProperty("defaultOp") ReduceOp defaultOp,
                    @JsonProperty("opMap") Map<String, List<ReduceOp>> opMap,
                    @JsonProperty("customReductions") Map<String, AggregableColumnReduction> customReductions,
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

            //First: check for a custom reduction on this column
            if (customReductions != null && customReductions.containsKey(name)) {
                AggregableColumnReduction reduction = customReductions.get(name);

                List<String> outName = reduction.getColumnsOutputName(name);
                List<ColumnMetaData> outMeta = reduction.getColumnOutputMetaData(outName, inMeta);
                newMeta.addAll(outMeta);
                continue;
            }

            //Second: check for conditional reductions on this column:
            if (conditionalReductions != null && conditionalReductions.containsKey(name)) {
                ConditionalReduction reduction = conditionalReductions.get(name);

                List<String> outNames = reduction.getOutputNames();
                List<ReduceOp> reductions = reduction.getReductions();
                for (int j = 0; j < reduction.getReductions().size(); j++) {
                    ReduceOp red = reductions.get(j);
                    String outName = outNames.get(j);
                    ColumnMetaData m = getMetaForColumn(red, name, inMeta);
                    m.setName(outName);
                    newMeta.add(m);
                }
                continue;
            }


            //Otherwise: get the specified (built-in) reduction op
            //If no reduction op is specified for that column: use the default
            List<ReduceOp> lop = opMap.containsKey(name) ? opMap.get(name) : Collections.singletonList(defaultOp);
            if (lop != null)
                for (ReduceOp op : lop) {
                    newMeta.add(getMetaForColumn(op, name, inMeta));
                }
        }

        return schema.newSchema(newMeta);
    }

    private static String getOutNameForColumn(ReduceOp op, String name) {
        return op.name().toLowerCase() + "(" + name + ")";
    }

    private static ColumnMetaData getMetaForColumn(ReduceOp op, String name, ColumnMetaData inMeta) {
        inMeta = inMeta.clone();
        switch (op) {
            // type-preserving operations
            case Min:
            case Max:
            case Range:
            case TakeFirst:
            case TakeLast:
                inMeta.setName(getOutNameForColumn(op, name));
                return inMeta;
            case Prod:
            case Sum:
                String outName = getOutNameForColumn(op, name);
                //Issue with prod/sum: the input meta data restrictions probably won't hold. But the data _type_ should essentially remain the same
                ColumnMetaData outMeta;
                if (inMeta instanceof IntegerMetaData)
                    outMeta = new IntegerMetaData(outName);
                else if (inMeta instanceof LongMetaData)
                    outMeta = new LongMetaData(outName);
                else if (inMeta instanceof FloatMetaData)
                    outMeta = new FloatMetaData(outName);
                else if (inMeta instanceof DoubleMetaData)
                    outMeta = new DoubleMetaData(outName);
                else { //Sum/Prod doesn't really make sense to sum other column types anyway...
                    outMeta = inMeta;
                }
                outMeta.setName(outName);
                return outMeta;
            case Mean:
            case Stdev:
            case Variance:
            case PopulationVariance:
            case UncorrectedStdDev:
                return new DoubleMetaData(getOutNameForColumn(op, name));
            case Append:
            case Prepend:
                return new StringMetaData(getOutNameForColumn(op, name));
            case Count: //Always Long
            case CountUnique:
                return new LongMetaData(getOutNameForColumn(op, name), 0L, null);
            default:
                throw new UnsupportedOperationException("Unknown or not implemented op: " + op);
        }
    }

    @Override
    public IAggregableReduceOp<List<Writable>, List<Writable>> aggregableReducer() {
        //Go through each writable, and reduce according to whatever strategy is specified

        if (schema == null)
            throw new IllegalStateException("Error: Schema has not been set");

        int nCols = schema.numColumns();
        List<String> colNames = schema.getColumnNames();

        List<IAggregableReduceOp<Writable, List<Writable>>> ops = new ArrayList<>(nCols);
        boolean conditionalActive = (conditionalReductions != null && !conditionalReductions.isEmpty());
        List<Condition> conditions = new ArrayList<>(nCols);

        for (int i = 0; i < nCols; i++) {
            String colName = colNames.get(i);
            if (keyColumnsSet != null && keyColumnsSet.contains(colName)) {
                IAggregableReduceOp<Writable, Writable> first = new AggregatorImpls.AggregableFirst<>();
                ops.add(new AggregableMultiOp<>(Collections.singletonList(first)));
                if (conditionalActive)
                    conditions.add(new TrivialColumnCondition(colName));
                continue;
            }


            // is this a *custom* reduction column?
            if (customReductions != null && customReductions.containsKey(colName)) {
                AggregableColumnReduction reduction = customReductions.get(colName);
                ops.add(reduction.reduceOp());
                if (conditionalActive)
                    conditions.add(new TrivialColumnCondition(colName));
                continue;
            }

            // are we adding global *conditional* reduction column?
            // Only practical difference with conditional reductions is we filter the input on an all-fields condition first
            if (conditionalActive) {
                if (conditionalReductions.containsKey(colName))
                    conditions.add(conditionalReductions.get(colName).getCondition());
                else
                    conditions.add(new TrivialColumnCondition(colName));
            }

            //What type of column is this?
            ColumnType type = schema.getType(i);

            //What ops are we performing on this column?
            boolean conditionalOp = conditionalActive && conditionalReductions.containsKey(colName);
            List<ReduceOp> lop =
                            (conditionalOp ? conditionalReductions.get(colName).getReductions() : opMap.get(colName));
            if (lop == null || lop.isEmpty())
                lop = Collections.singletonList(defaultOp);

            //Execute the reduction, store the result
            ops.add(AggregableReductionUtils.reduceColumn(lop, type, ignoreInvalidInColumns.contains(colName),
                            schema.getMetaData(i)));
        }

        if (conditionalActive) {
            return new DispatchWithConditionOp<>(ops, conditions);
        } else {
            return new DispatchOp<>(ops);
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Reducer(");
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
        private Map<String, List<ReduceOp>> opMap = new HashMap<>();
        private Map<String, AggregableColumnReduction> customReductions = new HashMap<>();
        private Map<String, ConditionalReduction> conditionalReductions = new HashMap<>();
        private Set<String> ignoreInvalidInColumns = new HashSet<>();
        private String[] keyColumns;


        /**
         * Create a Reducer builder, and set the default column reduction operation.
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
                List<ReduceOp> ops = new ArrayList<>();
                if (opMap.containsKey(s))
                    ops.addAll(opMap.get(s));
                ops.add(op);
                opMap.put(s, ops);
            }
            return this;
        }

        private Builder addAll(List<ReduceOp> ops, String[] cols) {
            for (String s : cols) {
                List<ReduceOp> theseOps = new ArrayList<>();
                if (opMap.containsKey(s))
                    theseOps.addAll(opMap.get(s));
                theseOps.addAll(ops);
                opMap.put(s, theseOps);
            }
            return this;
        }

        public Builder multipleOpColmumns(List<ReduceOp> ops, String... columns) {
            return addAll(ops, columns);
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
         * Reduce the specified columns by taking the product of values
         */
        public Builder prodColumns(String... columns) {
            return add(ReduceOp.Prod, columns);
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
         * Reduce the specified columns by taking the uncorrected standard deviation of the values
         */
        public Builder uncorrectedStdevColumns(String... columns) {
            return add(ReduceOp.Stdev, columns);
        }

        /**
         * Reduce the specified columns by taking the variance of the values
         */
        public Builder variance(String... columns) {
            return add(ReduceOp.Variance, columns);
        }

        /**
         * Reduce the specified columns by taking the population variance of the values
         */
        public Builder populationVariance(String... columns) {
            return add(ReduceOp.PopulationVariance, columns);
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
         * Reduce the specified columns by taking the concatenation of all content
         * Beware, the output will be huge!
         */
        public Builder appendColumns(String... columns) {
            return add(ReduceOp.Append, columns);
        }

        /**
         * Reduce the specified columns by taking the concatenation of all content in the reverse order
         * Beware, the output will be huge!
         */
        public Builder prependColumns(String... columns) {
            return add(ReduceOp.Prepend, columns);
        }

        /**
         * Reduce the specified column using a custom column reduction functionality.
         *
         * @param column          Column to execute the custom reduction functionality on
         * @param columnReduction Column reduction to execute on that column
         */
        public Builder customReduction(String column, AggregableColumnReduction columnReduction) {
            customReductions.put(column, columnReduction);
            return this;
        }

        /**
         * Conditional reduction: apply the reduces on a specified column, where the reduction occurs *only* on those
         * examples where the condition returns true. Examples where the condition does not apply (returns false) are
         * ignored/excluded.
         *
         * @param column     Name of the column to execute the conditional reduction on
         * @param outputName Name of the column, after the reduction has been executed
         * @param reductions  Reductions to execute
         * @param condition  Condition to use in the reductions
         */
        public Builder conditionalReduction(String column, List<String> outputNames, List<ReduceOp> reductions,
                        Condition condition) {
            Preconditions.checkArgument(outputNames.size() == reductions.size(),
                            "Conditional reductions should provide names for every column");
            this.conditionalReductions.put(column,
                            new ConditionalReduction(column, outputNames, reductions, condition));
            return this;
        }

        /**
         * Conditional reduction: apply the reduces on a specified column, where the reduction occurs *only* on those
         * examples where the condition returns true. Examples where the condition does not apply (returns false) are
         * ignored/excluded.
         *
         * @param column     Name of the column to execute the conditional reduction on
         * @param outputName Name of the column, after the reduction has been executed
         * @param reductions  Reductions to execute
         * @param condition  Condition to use in the reductions
         */
        public Builder conditionalReduction(String column, String outputName, ReduceOp reduction, Condition condition) {
            this.conditionalReductions.put(column, new ConditionalReduction(column,
                            Collections.singletonList(outputName), Collections.singletonList(reduction), condition));
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
        private final List<String> outputNames;
        private final List<ReduceOp> reductions;
        private final Condition condition;
    }

}
