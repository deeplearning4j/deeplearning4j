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

package org.datavec.api.transform;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import com.fasterxml.jackson.datatype.joda.JodaModule;
import org.datavec.api.transform.analysis.columns.ColumnAnalysis;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.transform.rank.CalculateSortedRank;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.sequence.ConvertFromSequence;
import org.datavec.api.transform.sequence.ConvertToSequence;
import org.datavec.api.transform.sequence.SequenceSplit;
import org.datavec.api.transform.sequence.window.ReduceSequenceByWindowTransform;
import org.datavec.api.transform.sequence.window.WindowFunction;
import org.datavec.api.transform.transform.categorical.CategoricalToIntegerTransform;
import org.datavec.api.transform.transform.categorical.IntegerToCategoricalTransform;
import org.datavec.api.transform.transform.categorical.StringToCategoricalTransform;
import org.datavec.api.transform.transform.column.*;
import org.datavec.api.transform.transform.condition.ConditionalCopyValueTransform;
import org.datavec.api.transform.transform.condition.ConditionalReplaceValueTransform;
import org.datavec.api.transform.transform.integer.IntegerColumnsMathOpTransform;
import org.datavec.api.transform.transform.longtransform.LongColumnsMathOpTransform;
import org.datavec.api.transform.transform.longtransform.LongMathOpTransform;
import org.datavec.api.transform.transform.normalize.Normalize;
import org.datavec.api.transform.transform.doubletransform.*;
import org.datavec.api.transform.transform.string.RemoveWhiteSpaceTransform;
import org.datavec.api.transform.transform.string.StringMapTransform;
import org.datavec.api.transform.transform.time.StringToTimeTransform;
import org.datavec.api.transform.transform.time.TimeMathOpTransform;
import org.datavec.api.transform.analysis.columns.NumericalColumnAnalysis;
import org.datavec.api.transform.sequence.SequenceComparator;
import org.datavec.api.transform.transform.categorical.CategoricalToOneHotTransform;
import lombok.Data;
import org.datavec.api.writable.Writable;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.reduce.IReducer;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.transform.transform.integer.IntegerMathOpTransform;
import org.datavec.api.writable.comparator.WritableComparator;
import org.joda.time.DateTimeZone;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * A TransformProcess defines an ordered list of transformations to be executed on some data
 *
 * @author Alex Black
 */
@Data
public class TransformProcess implements Serializable {

    private final Schema initialSchema;
    private List<DataAction> actionList;

    public TransformProcess(@JsonProperty("initialSchema") Schema initialSchema, @JsonProperty("actionList") List<DataAction> actionList){
        this.initialSchema = initialSchema;
        this.actionList = actionList;

        //Calculate and set the schemas for each tranformation:
        Schema currInputSchema = initialSchema;
        for (DataAction d : actionList) {
            if (d.getTransform() != null) {
                Transform t = d.getTransform();
                t.setInputSchema(currInputSchema);
                currInputSchema = t.transform(currInputSchema);
            } else if (d.getFilter() != null) {
                //Filter -> doesn't change schema. But we DO need to set the schema in the filter...
                d.getFilter().setInputSchema(currInputSchema);
            } else if (d.getConvertToSequence() != null) {
                if (currInputSchema instanceof SequenceSchema) {
                    throw new RuntimeException("Cannot convert to sequence: schema is already a sequence schema: " + currInputSchema);
                }
                ConvertToSequence cts = d.getConvertToSequence();
                cts.setInputSchema(currInputSchema);
                currInputSchema = cts.transform(currInputSchema);
            } else if (d.getConvertFromSequence() != null) {
                ConvertFromSequence cfs = d.getConvertFromSequence();
                if (!(currInputSchema instanceof SequenceSchema)) {
                    throw new RuntimeException("Cannot convert from sequence: schema is not a sequence schema: " + currInputSchema);
                }
                cfs.setInputSchema((SequenceSchema) currInputSchema);
                currInputSchema = cfs.transform((SequenceSchema) currInputSchema);
            } else if (d.getSequenceSplit() != null) {
                d.getSequenceSplit().setInputSchema(currInputSchema);
                continue;   //no change to sequence schema
            } else if (d.getReducer() != null) {
                IReducer reducer = d.getReducer();
                reducer.setInputSchema(currInputSchema);
                currInputSchema = reducer.transform(currInputSchema);
            } else if (d.getCalculateSortedRank() != null) {
                CalculateSortedRank csr = d.getCalculateSortedRank();
                csr.setInputSchema(currInputSchema);
                currInputSchema = csr.transform(currInputSchema);
            } else {
                throw new RuntimeException("Unknown action: " + d);
            }
        }
    }

    private TransformProcess(Builder builder) {
        this(builder.initialSchema, builder.actionList);
    }

    public List<DataAction> getActionList() {
        return actionList;
    }

    /**
     * Get the Schema of the output data, after executing the process
     *
     * @return Schema of the output data
     */
    public Schema getFinalSchema() {
        return getSchemaAfterStep(actionList.size());
    }

    /**
     * Return the schema after executing all steps up to and including the specified step.
     * Steps are indexed from 0: so getSchemaAfterStep(0) is after one transform has been executed.
     *
     * @param step Index of the step
     * @return Schema of the data, after that (and all prior) steps have been executed
     */
    public Schema getSchemaAfterStep(int step) {
        Schema currInputSchema = initialSchema;
        int i = 0;
        for (DataAction d : actionList) {
            if (d.getTransform() != null) {
                Transform t = d.getTransform();
                currInputSchema = t.transform(currInputSchema);
            } else if (d.getFilter() != null) {
                i++;
                continue; //Filter -> doesn't change schema
            } else if (d.getConvertToSequence() != null) {
                if (currInputSchema instanceof SequenceSchema) {
                    throw new RuntimeException("Cannot convert to sequence: schema is already a sequence schema: " + currInputSchema);
                }
                ConvertToSequence cts = d.getConvertToSequence();
                currInputSchema = cts.transform(currInputSchema);
            } else if (d.getConvertFromSequence() != null) {
                ConvertFromSequence cfs = d.getConvertFromSequence();
                if (!(currInputSchema instanceof SequenceSchema)) {
                    throw new RuntimeException("Cannot convert from sequence: schema is not a sequence schema: " + currInputSchema);
                }
                currInputSchema = cfs.transform((SequenceSchema) currInputSchema);
            } else if (d.getSequenceSplit() != null) {
                continue;   //Sequence split -> no change to schema
            } else if (d.getReducer() != null) {
                IReducer reducer = d.getReducer();
                currInputSchema = reducer.transform(currInputSchema);
            } else if (d.getCalculateSortedRank() != null) {
                CalculateSortedRank csr = d.getCalculateSortedRank();
                currInputSchema = csr.transform(currInputSchema);
            } else {
                throw new RuntimeException("Unknown action: " + d);
            }
            if (i++ == step) return currInputSchema;
        }
        return currInputSchema;
    }


    /**
     * Execute the full sequence of transformations for a single example. May return null if example is filtered
     * <b>NOTE:</b> Some TransformProcess operations cannot be done on examples individually. Most notably, ConvertToSequence
     * and ConvertFromSequence operations require the full data set to be processed at once
     *
     * @param input
     * @return
     */
    public List<Writable> execute(List<Writable> input) {
        List<Writable> currValues = input;

        for (DataAction d : actionList) {
            if (d.getTransform() != null) {
                Transform t = d.getTransform();
                currValues = t.map(currValues);

            } else if (d.getFilter() != null) {
                Filter f = d.getFilter();
                if (f.removeExample(currValues)) return null;
            } else if (d.getConvertToSequence() != null) {
                throw new RuntimeException("Cannot execute examples individually: TransformProcess contains a ConvertToSequence operation");
            } else if (d.getConvertFromSequence() != null) {
                throw new RuntimeException("Unexpected operation: TransformProcess contains a ConvertFromSequence operation");
            } else if (d.getSequenceSplit() != null) {
                throw new RuntimeException("Cannot execute examples individually: TransformProcess contains a SequenceSplit operation");
            } else {
                throw new RuntimeException("Unknown action: " + d);
            }
        }

        return currValues;
    }

    public List<List<Writable>> executeSequenceToSequence(List<List<Writable>> input) {
        List<List<Writable>> currValues = input;

        for (DataAction d : actionList) {
            if (d.getTransform() != null) {
                Transform t = d.getTransform();
                currValues = t.mapSequence(currValues);

            } else if (d.getFilter() != null) {
//                Filter f = d.getFilter();
//                if (f.removeExample(currValues)) return null;
                throw new RuntimeException("Sequence filtering not yet implemnted here");
            } else if (d.getConvertToSequence() != null) {
                throw new RuntimeException("Cannot execute examples individually: TransformProcess contains a ConvertToSequence operation");
            } else if (d.getConvertFromSequence() != null) {
                throw new RuntimeException("Unexpected operation: TransformProcess contains a ConvertFromSequence operation");
            } else if (d.getSequenceSplit() != null) {
                throw new RuntimeException("Cannot execute examples individually: TransformProcess contains a SequenceSplit operation");
            } else {
                throw new RuntimeException("Unknown action: " + d);
            }
        }

        return currValues;
    }

    /**
     * Execute the full sequence of transformations for a single time series (sequence). May return null if example is filtered
     */
    public List<List<Writable>> executeSequence(List<List<Writable>> inputSequence) {


        throw new UnsupportedOperationException("Not yet implemented");
    }


    public String toJson() {
        return toJacksonString(new JsonFactory());
    }

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

    public static TransformProcess fromJson(String json) {
        return fromJacksonString(json, new JsonFactory());
    }

    public static TransformProcess fromYaml(String yaml) {
        return fromJacksonString(yaml, new YAMLFactory());
    }

    private static TransformProcess fromJacksonString(String str, JsonFactory factory) {
        ObjectMapper om = new ObjectMapper(factory);
        om.registerModule(new JodaModule());
        om.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        om.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        om.enable(SerializationFeature.INDENT_OUTPUT);
        om.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.NONE);
        om.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
        try {
            return om.readValue(str, TransformProcess.class);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * Builder class for constructing a TransformProcess
     */
    public static class Builder {

        private List<DataAction> actionList = new ArrayList<>();
        private Schema initialSchema;

        public Builder(Schema initialSchema) {
            this.initialSchema = initialSchema;
        }

        /**
         * Add a transformation to be executed after the previously-added operations have been executed
         *
         * @param transform Transform to execute
         */
        public Builder transform(Transform transform) {
            actionList.add(new DataAction(transform));
            return this;
        }

        /**
         * Add a filter operation to be executed after the previously-added operations have been executed
         *
         * @param filter Filter operation to execute
         */
        public Builder filter(Filter filter) {
            actionList.add(new DataAction(filter));
            return this;
        }

        /**
         * Remove all of the specified columns, by name
         *
         * @param columnNames Names of the columns to remove
         */
        public Builder removeColumns(String... columnNames) {
            return transform(new RemoveColumnsTransform(columnNames));
        }

        /**
         * Remove all of the specified columns, by name
         *
         * @param columnNames Names of the columns to remove
         */
        public Builder removeColumns(Collection<String> columnNames) {
            return transform(new RemoveColumnsTransform(columnNames.toArray(new String[columnNames.size()])));
        }

        /**
         * Remove all columns, except for those that are specified here
         * @param columnNames    Names of the columns to keep
         */
        public Builder removeAllColumnsExceptFor(String... columnNames){
            return transform(new RemoveAllColumnsExceptForTransform(columnNames));
        }

        /**
         * Remove all columns, except for those that are specified here
         * @param columnNames    Names of the columns to keep
         */
        public Builder removeAllColumnsExceptFor(Collection<String> columnNames){
            return removeAllColumnsExceptFor(columnNames.toArray(new String[columnNames.size()]));
        }

        /**
         * Rename a single column
         *
         * @param oldName Original column name
         * @param newName New column name
         */
        public Builder renameColumn(String oldName, String newName) {
            return transform(new RenameColumnsTransform(oldName, newName));
        }

        /**
         * Rename multiple columns
         *
         * @param oldNames List of original column names
         * @param newNames List of new column names
         */
        public Builder renameColumns(List<String> oldNames, List<String> newNames) {
            return transform(new RenameColumnsTransform(oldNames, newNames));
        }

        /**
         * Reorder the columns using a partial or complete new ordering.
         * If only some of the column names are specified for the new order, the remaining columns will be placed at
         * the end, according to their current relative ordering
         *
         * @param newOrder Names of the columns, in the order they will appear in the output
         */
        public Builder reorderColumns(String... newOrder) {
            return transform(new ReorderColumnsTransform(newOrder));
        }

        /**
         * Duplicate a single column
         *
         * @param column Name of the column to duplicate
         * @param newName    Name of the new (duplicate) column
         */
        public Builder duplicateColumn(String column, String newName) {
            return transform(new DuplicateColumnsTransform(Collections.singletonList(column), Collections.singletonList(newName)));
        }


        /**
         * Duplicate a set of columns
         *
         * @param columnNames Names of the columns to duplicate
         * @param newNames    Names of the new (duplicated) columns
         */
        public Builder duplicateColumns(List<String> columnNames, List<String> newNames) {
            return transform(new DuplicateColumnsTransform(columnNames, newNames));
        }

        /**
         * Perform a mathematical operation (add, subtract, scalar max etc) on the specified integer column, with a scalar
         *
         * @param column The integer column to perform the operation on
         * @param mathOp     The mathematical operation
         * @param scalar     The scalar value to use in the mathematical operation
         */
        public Builder integerMathOp(String column, MathOp mathOp, int scalar) {
            return transform(new IntegerMathOpTransform(column, mathOp, scalar));
        }

        /**
         * Calculate and add a new integer column by performing a mathematical operation on a number of existing columns.
         * New column is added to the end.
         *
         * @param newColumnName Name of the new/derived column
         * @param mathOp        Mathematical operation to execute on the columns
         * @param columnNames   Names of the columns to use in the mathematical operation
         */
        public Builder integerColumnsMathOp(String newColumnName, MathOp mathOp, String... columnNames) {
            return transform(new IntegerColumnsMathOpTransform(newColumnName, mathOp, columnNames));
        }

        /**
         * Perform a mathematical operation (add, subtract, scalar max etc) on the specified long column, with a scalar
         *
         * @param columnName The long column to perform the operation on
         * @param mathOp     The mathematical operation
         * @param scalar     The scalar value to use in the mathematical operation
         */
        public Builder longMathOp(String columnName, MathOp mathOp, long scalar) {
            return transform(new LongMathOpTransform(columnName, mathOp, scalar));
        }

        /**
         * Calculate and add a new long column by performing a mathematical operation on a number of existing columns.
         * New column is added to the end.
         *
         * @param newColumnName Name of the new/derived column
         * @param mathOp        Mathematical operation to execute on the columns
         * @param columnNames   Names of the columns to use in the mathematical operation
         */
        public Builder longColumnsMathOp(String newColumnName, MathOp mathOp, String... columnNames) {
            return transform(new LongColumnsMathOpTransform(newColumnName, mathOp, columnNames));
        }

        /**
         * Perform a mathematical operation (add, subtract, scalar max etc) on the specified double column, with a scalar
         *
         * @param columnName The double column to perform the operation on
         * @param mathOp     The mathematical operation
         * @param scalar     The scalar value to use in the mathematical operation
         */
        public Builder doubleMathOp(String columnName, MathOp mathOp, double scalar) {
            return transform(new DoubleMathOpTransform(columnName, mathOp, scalar));
        }

        /**
         * Calculate and add a new double column by performing a mathematical operation on a number of existing columns.
         * New column is added to the end.
         *
         * @param newColumnName Name of the new/derived column
         * @param mathOp        Mathematical operation to execute on the columns
         * @param columnNames   Names of the columns to use in the mathematical operation
         */
        public Builder doubleColumnsMathOp(String newColumnName, MathOp mathOp, String... columnNames) {
            return transform(new DoubleColumnsMathOpTransform(newColumnName, mathOp, columnNames));
        }

        /**
         * Perform a mathematical operation (add, subtract, scalar min/max only) on the specified time column
         *
         * @param columnName   The integer column to perform the operation on
         * @param mathOp       The mathematical operation
         * @param timeQuantity The quantity used in the mathematical op
         * @param timeUnit     The unit that timeQuantity is specified in
         */
        public Builder timeMathOp(String columnName, MathOp mathOp, long timeQuantity, TimeUnit timeUnit) {
            return transform(new TimeMathOpTransform(columnName, mathOp, timeQuantity, timeUnit));
        }


        /**
         * Convert the specified column(s) from a categorical representation to a one-hot representation.
         * This involves the creation of multiple new columns each.
         *
         * @param columnNames Names of the categorical column(s) to convert to a one-hot representation
         */
        public Builder categoricalToOneHot(String... columnNames) {
            for (String s : columnNames) {
                transform(new CategoricalToOneHotTransform(s));
            }
            return this;
        }

        /**
         * Convert the specified column(s) from a categorical representation to an integer representation.
         * This will replace the specified categorical column(s) with an integer repreesentation, where
         * each integer has the value 0 to numCategories-1.
         *
         * @param columnNames Name of the categorical column(s) to convert to an integer representation
         */
        public Builder categoricalToInteger(String... columnNames) {
            for (String s : columnNames) {
                transform(new CategoricalToIntegerTransform(s));
            }
            return this;
        }

        /**
         * Convert the specified column from an integer representation (assume values 0 to numCategories-1) to
         * a categorical representation, given the specified state names
         *
         * @param columnName         Name of the column to convert
         * @param categoryStateNames Names of the states for the categorical column
         */
        public Builder integerToCategorical(String columnName, List<String> categoryStateNames) {
            return transform(new IntegerToCategoricalTransform(columnName, categoryStateNames));
        }

        /**
         * Convert the specified column from an integer representation to a categorical representation, given the specified
         * mapping between integer indexes and state names
         *
         * @param columnName           Name of the column to convert
         * @param categoryIndexNameMap Names of the states for the categorical column
         */
        public Builder integerToCategorical(String columnName, Map<Integer, String> categoryIndexNameMap) {
            return transform(new IntegerToCategoricalTransform(columnName, categoryIndexNameMap));
        }

        /**
         * Normalize the specified column with a given type of normalization
         *
         * @param column Column to normalize
         * @param type   Type of normalization to apply
         * @param da     DataAnalysis object
         */
        public Builder normalize(String column, Normalize type, DataAnalysis da) {

            ColumnAnalysis ca = da.getColumnAnalysis(column);
            if (!(ca instanceof NumericalColumnAnalysis))
                throw new IllegalStateException("Column \"" + column + "\" analysis is not numerical. "
                        + "Column is not numerical?");

            NumericalColumnAnalysis nca = (NumericalColumnAnalysis) ca;
            double min = nca.getMinDouble();
            double max = nca.getMaxDouble();
            double mean = nca.getMean();
            double sigma = nca.getSampleStdev();

            switch (type) {
                case MinMax:
                    return transform(new MinMaxNormalizer(column, min, max));
                case MinMax2:
                    return transform(new MinMaxNormalizer(column, min, max, -1, 1));
                case Standardize:
                    return transform(new StandardizeNormalizer(column, mean, sigma));
                case SubtractMean:
                    return transform(new SubtractMeanNormalizer(column, mean));
                case Log2Mean:
                    return transform(new Log2Normalizer(column, mean, min, 0.5));
                case Log2MeanExcludingMin:
                    long countMin = nca.getCountMinValue();

                    //mean including min value: (sum/totalCount)
                    //mean excluding min value: (sum - countMin*min)/(totalCount - countMin)
                    double meanExMin = (mean * ca.getCountTotal() - countMin * min) / (ca.getCountTotal() - countMin);
                    return transform(new Log2Normalizer(column, meanExMin, min, 0.5));
                default:
                    throw new RuntimeException("Unknown/not implemented normalization type: " + type);
            }
        }

        /**
         * Convert a set of independent records/examples into a sequence, according to some key.
         * Within each sequence, values are ordered using the provided {@link SequenceComparator}
         *
         * @param keyColumn  Column to use as a key (values with the same key will be combined into sequences)
         * @param comparator A SequenceComparator to order the values within each sequence (for example, by time or String order)
         */
        public Builder convertToSequence(String keyColumn, SequenceComparator comparator) {
            actionList.add(new DataAction(new ConvertToSequence(keyColumn, comparator)));
            return this;
        }


        /**
         * Convert a sequence to a set of individual values (by treating each value in each sequence as a separate example)
         */
        public Builder convertFromSequence() {
            actionList.add(new DataAction(new ConvertFromSequence()));
            return this;
        }

        /**
         * Split sequences into 1 or more other sequences. Used for example to split large sequences into a set of smaller sequences
         *
         * @param split SequenceSplit that defines how splits will occur
         */
        public Builder splitSequence(SequenceSplit split) {
            actionList.add(new DataAction(split));
            return this;
        }

        /**
         * Reduce (i.e., aggregate/combine) a set of examples (typically by key).
         * <b>Note</b>: In the current implementation, reduction operations can be performed only on standard (i.e., non-sequence) data
         *
         * @param reducer Reducer to use
         */
        public Builder reduce(IReducer reducer) {
            actionList.add(new DataAction(reducer));
            return this;
        }

        /**
         * Reduce (i.e., aggregate/combine) a set of sequence examples - for each sequence individually - using a window function.
         * For example, take all records/examples in each 24-hour period (i.e., using window function), and convert them into
         * a singe value (using the reducer). In this example, the output is a sequence, with time period of 24 hours.
         *
         * @param reducer        Reducer to use to reduce each window
         * @param windowFunction Window function to find apply on each sequence individually
         */
        public Builder reduceSequenceByWindow(IReducer reducer, WindowFunction windowFunction) {
            actionList.add(new DataAction(new ReduceSequenceByWindowTransform(reducer, windowFunction)));
            return this;
        }

        /**
         * CalculateSortedRank: calculate the rank of each example, after sorting example.
         * For example, we might have some numerical "score" column, and we want to know for the rank (sort order) for each
         * example, according to that column.<br>
         * The rank of each example (after sorting) will be added in a new Long column. Indexing is done from 0; examples will have
         * values 0 to dataSetSize-1.<br>
         * <p>
         * Currently, CalculateSortedRank can only be applied on standard (i.e., non-sequence) data
         * Furthermore, the current implementation can only sort on one column
         *
         * @param newColumnName Name of the new column (will contain the rank for each example)
         * @param sortOnColumn  Column to sort on
         * @param comparator    Comparator used to sort examples
         */
        public Builder calculateSortedRank(String newColumnName, String sortOnColumn, WritableComparator comparator) {
            actionList.add(new DataAction(new CalculateSortedRank(newColumnName, sortOnColumn, comparator)));
            return this;
        }

        /**
         * CalculateSortedRank: calculate the rank of each example, after sorting example.
         * For example, we might have some numerical "score" column, and we want to know for the rank (sort order) for each
         * example, according to that column.<br>
         * The rank of each example (after sorting) will be added in a new Long column. Indexing is done from 0; examples will have
         * values 0 to dataSetSize-1.<br>
         * <p>
         * Currently, CalculateSortedRank can only be applied on standard (i.e., non-sequence) data
         * Furthermore, the current implementation can only sort on one column
         *
         * @param newColumnName Name of the new column (will contain the rank for each example)
         * @param sortOnColumn  Column to sort on
         * @param comparator    Comparator used to sort examples
         * @param ascending     If true: sort ascending. False: descending
         */
        public Builder calculateSortedRank(String newColumnName, String sortOnColumn, WritableComparator comparator, boolean ascending) {
            actionList.add(new DataAction(new CalculateSortedRank(newColumnName, sortOnColumn, comparator, ascending)));
            return this;
        }

        /**
         * Convert the specified String column to a categorical column. The state names must be provided.
         *
         * @param columnName Name of the String column to convert to categorical
         * @param stateNames State names of the category
         */
        public Builder stringToCategorical(String columnName, List<String> stateNames) {
            return transform(new StringToCategoricalTransform(columnName, stateNames));
        }

        /**
         * Remove all whitespace characters from the values in the specified String column
         *
         * @param columnName Name of the column to remove whitespace from
         */
        public Builder stringRemoveWhitespaceTransform(String columnName) {
            return transform(new RemoveWhiteSpaceTransform(columnName));
        }

        /**
         * Replace one or more String values in the specified column with new values.
         * <p>
         * Keys in the map are the original values; the Values in the map are their replacements.
         * If a String appears in the data but does not appear in the provided map (as a key), that String values will
         * not be modified.
         *
         * @param columnName Name of the column in which to do replacement
         * @param mapping    Map of oldValues -> newValues
         */
        public Builder stringMapTransform(String columnName, Map<String, String> mapping) {
            return transform(new StringMapTransform(columnName, mapping));
        }

        /**
         * Convert a String column (containing a date/time String) to a time column (by parsing the date/time String)
         *
         * @param column       String column containing the date/time Strings
         * @param format       Format of the strings. Time format is specified as per http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html
         * @param dateTimeZone Timezone of the column
         */
        public Builder stringToTimeTransform(String column, String format, DateTimeZone dateTimeZone) {
            return transform(new StringToTimeTransform(column, format, dateTimeZone));
        }

        /**
         * Replace the values in a specified column with a specified new value, if some condition holds.
         * If the condition does not hold, the original values are not modified.
         *
         * @param column    Column to operate on
         * @param newValue  Value to use as replacement, if condition is satisfied
         * @param condition Condition that must be satisfied for replacement
         */
        public Builder conditionalReplaceValueTransform(String column, Writable newValue, Condition condition) {
            return transform(new ConditionalReplaceValueTransform(column, newValue, condition));
        }

        /**
         * Replace the value in a specified column with a new value taken from another column, if a condition is satisfied/true.<br>
         * Note that the condition can be any generic condition, including on other column(s), different to the column
         * that will be modified if the condition is satisfied/true.<br>
         *
         * @param columnToReplace    Name of the column in which values will be replaced (if condition is satisfied)
         * @param sourceColumn       Name of the column from which the new values will be
         * @param condition          Condition to use
         */
        public Builder conditionalCopyValueTransform(String columnToReplace, String sourceColumn, Condition condition) {
            return transform(new ConditionalCopyValueTransform(columnToReplace, sourceColumn, condition));
        }

        /**
         * Create the TransformProcess object
         */
        public TransformProcess build() {
            return new TransformProcess(this);
        }
    }


}
