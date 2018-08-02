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

package org.datavec.api.transform;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.analysis.columns.ColumnAnalysis;
import org.datavec.api.transform.analysis.columns.NumericalColumnAnalysis;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.transform.ndarray.NDArrayColumnsMathOpTransform;
import org.datavec.api.transform.ndarray.NDArrayDistanceTransform;
import org.datavec.api.transform.ndarray.NDArrayMathFunctionTransform;
import org.datavec.api.transform.ndarray.NDArrayScalarOpTransform;
import org.datavec.api.transform.rank.CalculateSortedRank;
import org.datavec.api.transform.reduce.IAssociativeReducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.transform.sequence.*;
import org.datavec.api.transform.sequence.trim.SequenceTrimTransform;
import org.datavec.api.transform.sequence.window.ReduceSequenceByWindowTransform;
import org.datavec.api.transform.sequence.window.WindowFunction;
import org.datavec.api.transform.serde.JsonMappers;
import org.datavec.api.transform.transform.categorical.CategoricalToIntegerTransform;
import org.datavec.api.transform.transform.categorical.CategoricalToOneHotTransform;
import org.datavec.api.transform.transform.categorical.IntegerToCategoricalTransform;
import org.datavec.api.transform.transform.categorical.StringToCategoricalTransform;
import org.datavec.api.transform.transform.column.*;
import org.datavec.api.transform.transform.condition.ConditionalCopyValueTransform;
import org.datavec.api.transform.transform.condition.ConditionalReplaceValueTransform;
import org.datavec.api.transform.transform.condition.ConditionalReplaceValueTransformWithDefault;
import org.datavec.api.transform.transform.doubletransform.*;
import org.datavec.api.transform.transform.floattransform.FloatColumnsMathOpTransform;
import org.datavec.api.transform.transform.floattransform.FloatMathFunctionTransform;
import org.datavec.api.transform.transform.floattransform.FloatMathOpTransform;
import org.datavec.api.transform.transform.integer.ConvertToInteger;
import org.datavec.api.transform.transform.integer.IntegerColumnsMathOpTransform;
import org.datavec.api.transform.transform.integer.IntegerMathOpTransform;
import org.datavec.api.transform.transform.integer.IntegerToOneHotTransform;
import org.datavec.api.transform.transform.longtransform.LongColumnsMathOpTransform;
import org.datavec.api.transform.transform.longtransform.LongMathOpTransform;
import org.datavec.api.transform.transform.normalize.Normalize;
import org.datavec.api.transform.transform.sequence.SequenceMovingWindowReduceTransform;
import org.datavec.api.transform.transform.sequence.SequenceOffsetTransform;
import org.datavec.api.transform.transform.string.*;
import org.datavec.api.transform.transform.time.StringToTimeTransform;
import org.datavec.api.transform.transform.time.TimeMathOpTransform;
import org.datavec.api.writable.*;
import org.datavec.api.writable.comparator.WritableComparator;
import org.joda.time.DateTimeZone;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * A TransformProcess defines
 * an ordered list of transformations
 * to be executed on some data
 *
 * @author Alex Black
 */
@Data
@Slf4j
public class TransformProcess implements Serializable {

    private final Schema initialSchema;
    private List<DataAction> actionList;

    public TransformProcess(@JsonProperty("initialSchema") Schema initialSchema,
                            @JsonProperty("actionList") List<DataAction> actionList) {
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
                    throw new RuntimeException("Cannot convert to sequence: schema is already a sequence schema: "
                            + currInputSchema);
                }
                ConvertToSequence cts = d.getConvertToSequence();
                cts.setInputSchema(currInputSchema);
                currInputSchema = cts.transform(currInputSchema);
            } else if (d.getConvertFromSequence() != null) {
                ConvertFromSequence cfs = d.getConvertFromSequence();
                if (!(currInputSchema instanceof SequenceSchema)) {
                    throw new RuntimeException("Cannot convert from sequence: schema is not a sequence schema: "
                            + currInputSchema);
                }
                cfs.setInputSchema((SequenceSchema) currInputSchema);
                currInputSchema = cfs.transform((SequenceSchema) currInputSchema);
            } else if (d.getSequenceSplit() != null) {
                d.getSequenceSplit().setInputSchema(currInputSchema);
                continue; //no change to sequence schema
            } else if (d.getReducer() != null) {
                IAssociativeReducer reducer = d.getReducer();
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

    /**
     * Get the action list that this transform process
     * will execute
     * @return
     */
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
                    throw new RuntimeException("Cannot convert to sequence: schema is already a sequence schema: "
                            + currInputSchema);
                }
                ConvertToSequence cts = d.getConvertToSequence();
                currInputSchema = cts.transform(currInputSchema);
            } else if (d.getConvertFromSequence() != null) {
                ConvertFromSequence cfs = d.getConvertFromSequence();
                if (!(currInputSchema instanceof SequenceSchema)) {
                    throw new RuntimeException("Cannot convert from sequence: schema is not a sequence schema: "
                            + currInputSchema);
                }
                currInputSchema = cfs.transform((SequenceSchema) currInputSchema);
            } else if (d.getSequenceSplit() != null) {
                continue; //Sequence split -> no change to schema
            } else if (d.getReducer() != null) {
                IAssociativeReducer reducer = d.getReducer();
                currInputSchema = reducer.transform(currInputSchema);
            } else if (d.getCalculateSortedRank() != null) {
                CalculateSortedRank csr = d.getCalculateSortedRank();
                currInputSchema = csr.transform(currInputSchema);
            } else {
                throw new RuntimeException("Unknown action: " + d);
            }
            if (i++ == step)
                return currInputSchema;
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
                if (f.removeExample(currValues))
                    return null;
            } else if (d.getConvertToSequence() != null) {
                throw new RuntimeException(
                        "Cannot execute examples individually: TransformProcess contains a ConvertToSequence operation");
            } else if (d.getConvertFromSequence() != null) {
                throw new RuntimeException(
                        "Unexpected operation: TransformProcess contains a ConvertFromSequence operation");
            } else if (d.getSequenceSplit() != null) {
                throw new RuntimeException(
                        "Cannot execute examples individually: TransformProcess contains a SequenceSplit operation");
            } else {
                throw new RuntimeException("Unknown action: " + d);
            }
        }

        return currValues;
    }

    /**
     *
     * @param input
     * @return
     */
    public List<List<Writable>> executeSequenceToSequence(List<List<Writable>> input) {
        List<List<Writable>> currValues = input;

        for (DataAction d : actionList) {
            if (d.getTransform() != null) {
                Transform t = d.getTransform();
                currValues = t.mapSequence(currValues);

            } else if (d.getFilter() != null) {
                if (d.getFilter().removeSequence(currValues)) {
                    return null;
                }
            } else if (d.getConvertToSequence() != null) {
                throw new RuntimeException(
                        "Cannot execute examples individually: TransformProcess contains a ConvertToSequence operation");
            } else if (d.getConvertFromSequence() != null) {
                throw new RuntimeException(
                        "Unexpected operation: TransformProcess contains a ConvertFromSequence operation");
            } else if (d.getSequenceSplit() != null) {
                throw new RuntimeException(
                        "Cannot execute examples individually: TransformProcess contains a SequenceSplit operation");
            } else {
                throw new RuntimeException("Unknown or not supported action: " + d);
            }
        }

        return currValues;
    }

    /**
     * Execute the full sequence of transformations for a single time series (sequence). May return null if example is filtered
     */
    public List<List<Writable>> executeSequence(List<List<Writable>> inputSequence) {
        return executeSequenceToSequence(inputSequence);
    }


    /**
     * Execute a TransformProcess that starts with a single (non-sequence) record,
     * and converts it to a sequence record.
     * <b>NOTE</b>: This method has the following significant limitation:
     * if it contains a ConvertToSequence op,
     * it MUST be using singleStepSequencesMode - see {@link ConvertToSequence} for details.<br>
     * This restriction is necessary, as ConvertToSequence.singleStepSequencesMode is false, this requires a group by
     * operation - i.e., we need to group multiple independent records together by key(s) - this isn't possible here,
     * when providing a single example as input
     *
     * @param inputExample Input example
     * @return Sequence, after processing (or null, if it was filtered out)
     */
    public List<List<List<Writable>>> executeToSequenceBatch(List<List<Writable>> inputExample){
        List<List<List<Writable>>> ret = new ArrayList<>();
        for(List<Writable> record : inputExample)
            ret.add(execute(record, null).getRight());
        return ret;
    }

    /**
     * Execute a TransformProcess that starts with a single (non-sequence) record,
     * and converts it to a sequence record.
     * <b>NOTE</b>: This method has the following significant limitation:
     * if it contains a ConvertToSequence op,
     * it MUST be using singleStepSequencesMode - see {@link ConvertToSequence} for details.<br>
     * This restriction is necessary, as ConvertToSequence.singleStepSequencesMode is false, this requires a group by
     * operation - i.e., we need to group multiple independent records together by key(s) - this isn't possible here,
     * when providing a single example as input
     *
     * @param inputExample Input example
     * @return Sequence, after processing (or null, if it was filtered out)
     */
    public List<List<Writable>> executeToSequence(List<Writable> inputExample){
        return execute(inputExample, null).getRight();
    }

    /**
     * Execute a TransformProcess that starts with a sequence
     * record, and converts it to a single (non-sequence) record
     *
     * @param inputSequence Input sequence
     * @return Record after processing (or null if filtered out)
     */
    public List<Writable> executeSequenceToSingle(List<List<Writable>> inputSequence){
        return execute(null, inputSequence).getLeft();
    }

    private Pair<List<Writable>, List<List<Writable>>> execute(List<Writable> currEx, List<List<Writable>> currSeq){
        for (DataAction d : actionList) {
            if (d.getTransform() != null) {
                Transform t = d.getTransform();

                if(currEx != null){
                    currEx = t.map(currEx);
                    currSeq = null;
                } else {
                    currEx = null;
                    currSeq = t.mapSequence(currSeq);
                }
            } else if (d.getFilter() != null) {
                if( (currEx != null && d.getFilter().removeExample(currEx)) || d.getFilter().removeSequence(currEx)){
                    return new Pair<>(null, null);
                }
            } else if (d.getConvertToSequence() != null) {

                if(d.getConvertToSequence().isSingleStepSequencesMode()){
                    if(currSeq != null){
                        throw new RuntimeException("Cannot execute ConvertToSequence op: current records are already a sequence");
                    } else {
                        currSeq = Collections.singletonList(currEx);
                        currEx = null;
                    }
                } else {
                    //Can't execute this - would require a group-by operation, and we only have 1 example!
                    throw new RuntimeException( "Cannot execute examples individually: TransformProcess contains a" +
                            " ConvertToSequence operation, with singleStepSequnceeMode == false. Only " +
                            " ConvertToSequence operations with singleStepSequnceeMode == true can be executed individually " +
                            "as other types require a groupBy operation (which cannot be executed when only a sinlge record) " +
                            "is provided as input");
                }
            } else if (d.getConvertFromSequence() != null) {
                throw new RuntimeException("Unexpected operation: TransformProcess contains a ConvertFromSequence" +
                        " operation. This would produce multiple output records, which cannot be executed using this method");
            } else if (d.getSequenceSplit() != null) {
                throw new RuntimeException( "Cannot execute examples individually: TransformProcess contains a" +
                        " SequenceSplit operation. This would produce multiple output records, which cannot be executed" +
                        " using this method");
            } else {
                throw new RuntimeException("Unknown or not supported action: " + d);
            }
        }

        return new Pair<>(currEx, currSeq);
    }

    /**
     * Convert the TransformProcess to a JSON string
     *
     * @return TransformProcess, as JSON
     */
    public String toJson() {
        try {
            return JsonMappers.getMapper().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            //TODO proper exception message
            throw new RuntimeException(e);
        }
    }

    /**
     * Convert the TransformProcess to a YAML string
     *
     * @return TransformProcess, as YAML
     */
    public String toYaml() {
        try {
            return JsonMappers.getMapper().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            //TODO proper exception message
            throw new RuntimeException(e);
        }
    }

    /**
     * Deserialize a JSON String (created by {@link #toJson()}) to a TransformProcess
     *
     * @return TransformProcess, from JSON
     */
    public static TransformProcess fromJson(String json) {
        try {
            return JsonMappers.getMapper().readValue(json, TransformProcess.class);
        } catch (IOException e) {
            //TODO proper exception message
            throw new RuntimeException(e);
        }
    }

    /**
     * Deserialize a JSON String (created by {@link #toJson()}) to a TransformProcess
     *
     * @return TransformProcess, from JSON
     */
    public static TransformProcess fromYaml(String yaml) {
        try {
            return JsonMappers.getMapper().readValue(yaml, TransformProcess.class);
        } catch (IOException e) {
            //TODO proper exception message
            throw new RuntimeException(e);
        }
    }


    /**
     * Infer the categories for the given record reader for a particular column
     *  Note that each "column index" is a column in the context of:
     * List<Writable> record = ...;
     * record.get(columnIndex);
     *
     *  Note that anything passed in as a column will be automatically converted to a
     *  string for categorical purposes.
     *
     *  The *expected* input is strings or numbers (which have sensible toString() representations)
     *
     *  Note that the returned categories will be sorted alphabetically
     *
     * @param recordReader the record reader to iterate through
     * @param columnIndex te column index to get categories for
     * @return
     */
    public static List<String> inferCategories(RecordReader recordReader,int columnIndex) {
        Set<String> categories = new HashSet<>();
        while(recordReader.hasNext()) {
            List<Writable> next = recordReader.next();
            categories.add(next.get(columnIndex).toString());
        }

        //Sort categories alphabetically - HashSet and RecordReader orders are not deterministic in general
        List<String> ret = new ArrayList<>(categories);
        Collections.sort(ret);
        return ret;
    }

    /**
     * Infer the categories for the given record reader for
     * a particular set of columns (this is more efficient than
     * {@link #inferCategories(RecordReader, int)}
     * if you have more than one column you plan on inferring categories for)
     *
     * Note that each "column index" is a column in the context of:
     * List<Writable> record = ...;
     * record.get(columnIndex);
     *
     *
     *  Note that anything passed in as a column will be automatically converted to a
     *  string for categorical purposes. Results may vary depending on what's passed in.
     *  The *expected* input is strings or numbers (which have sensible toString() representations)
     *
     * Note that the returned categories will be sorted alphabetically, for each column
     *
     * @param recordReader the record reader to scan
     * @param columnIndices the column indices the get
     * @return the inferred categories
     */
    public static Map<Integer,List<String>> inferCategories(RecordReader recordReader,int[] columnIndices) {
        if(columnIndices == null || columnIndices.length < 1) {
            return Collections.emptyMap();
        }

        Map<Integer,List<String>> categoryMap = new HashMap<>();
        Map<Integer,Set<String>> categories = new HashMap<>();
        for(int i = 0; i < columnIndices.length; i++) {
            categoryMap.put(columnIndices[i],new ArrayList<String>());
            categories.put(columnIndices[i],new HashSet<String>());
        }
        while(recordReader.hasNext()) {
            List<Writable> next = recordReader.next();
            for(int i = 0; i < columnIndices.length; i++) {
                if(columnIndices[i] >= next.size()) {
                    log.warn("Filtering out example: Invalid length of columns");
                    continue;
                }

                categories.get(columnIndices[i]).add(next.get(columnIndices[i]).toString());
            }

        }

        for(int i = 0; i < columnIndices.length; i++) {
            categoryMap.get(columnIndices[i]).addAll(categories.get(columnIndices[i]));

            //Sort categories alphabetically - HashSet and RecordReader orders are not deterministic in general
            Collections.sort(categoryMap.get(columnIndices[i]));
        }

        return categoryMap;
    }

    /**
     * Transforms a sequence
     * of strings in to a sequence of writables
     * (very similar to {@link #transformRawStringsToInput(String...)}
     * for sequences
     * @param sequence the sequence to transform
     * @return the transformed input
     */
    public List<List<Writable>> transformRawStringsToInputSequence(List<List<String>> sequence) {
        List<List<Writable>> ret = new ArrayList<>();
        for(List<String> input : sequence)
            ret.add(transformRawStringsToInputList(input));
        return ret;
    }


    /**
     * Based on the input schema,
     * map raw string values to the appropriate
     * writable
     * @param values the values to convert
     * @return the transformed values based on the schema
     */
    public List<Writable> transformRawStringsToInputList(List<String> values) {
        List<Writable> ret = new ArrayList<>();
        if (values.size() != initialSchema.numColumns())
            throw new IllegalArgumentException(
                    String.format("Number of values %d does not match the number of input columns %d for schema",
                            values.size(), initialSchema.numColumns()));
        for (int i = 0; i < values.size(); i++) {
            switch (initialSchema.getType(i)) {
                case String:
                    ret.add(new Text(values.get(i)));
                    break;
                case Integer:
                    ret.add(new IntWritable(Integer.parseInt(values.get(i))));
                    break;
                case Double:
                    ret.add(new DoubleWritable(Double.parseDouble(values.get(i))));
                    break;
                case Float:
                    ret.add(new FloatWritable(Float.parseFloat(values.get(i))));
                    break;
                case Categorical:
                    ret.add(new Text(values.get(i)));
                    break;
                case Boolean:
                    ret.add(new BooleanWritable(Boolean.parseBoolean(values.get(i))));
                    break;
                case Time:

                    break;
                case Long:
                    ret.add(new LongWritable(Long.parseLong(values.get(i))));
            }
        }
        return ret;
    }


    /**
     * Based on the input schema,
     * map raw string values to the appropriate
     * writable
     * @param values the values to convert
     * @return the transformed values based on the schema
     */
    public List<Writable> transformRawStringsToInput(String... values) {
        return transformRawStringsToInputList(Arrays.asList(values));
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
         * Add a filter operation, based on the specified condition.
         *
         * If condition is satisfied (returns true): remove the example or sequence<br>
         * If condition is not satisfied (returns false): keep the example or sequence
         *
         * @param condition Condition to filter on
         */
        public Builder filter(Condition condition) {
            return filter(new ConditionFilter(condition));
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
        public Builder removeAllColumnsExceptFor(String... columnNames) {
            return transform(new RemoveAllColumnsExceptForTransform(columnNames));
        }

        /**
         * Remove all columns, except for those that are specified here
         * @param columnNames    Names of the columns to keep
         */
        public Builder removeAllColumnsExceptFor(Collection<String> columnNames) {
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
            return transform(new DuplicateColumnsTransform(Collections.singletonList(column),
                    Collections.singletonList(newName)));
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
         * @param columnName The float column to perform the operation on
         * @param mathOp     The mathematical operation
         * @param scalar     The scalar value to use in the mathematical operation
         */
        public Builder floatMathOp(String columnName, MathOp mathOp, float scalar) {
            return transform(new FloatMathOpTransform(columnName, mathOp, scalar));
        }

        /**
         * Calculate and add a new float column by performing a mathematical operation on a number of existing columns.
         * New column is added to the end.
         *
         * @param newColumnName Name of the new/derived column
         * @param mathOp        Mathematical operation to execute on the columns
         * @param columnNames   Names of the columns to use in the mathematical operation
         */
        public Builder floatColumnsMathOp(String newColumnName, MathOp mathOp, String... columnNames) {
            return transform(new FloatColumnsMathOpTransform(newColumnName, mathOp, columnNames));
        }

        /**
         * Perform a mathematical operation (such as sin(x), ceil(x), exp(x) etc) on a column
         *
         * @param columnName   Column name to operate on
         * @param mathFunction MathFunction to apply to the column
         */
        public Builder floatMathFunction(String columnName, MathFunction mathFunction) {
            return transform(new FloatMathFunctionTransform(columnName, mathFunction));
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
         * Perform a mathematical operation (such as sin(x), ceil(x), exp(x) etc) on a column
         *
         * @param columnName   Column name to operate on
         * @param mathFunction MathFunction to apply to the column
         */
        public Builder doubleMathFunction(String columnName, MathFunction mathFunction) {
            return transform(new DoubleMathFunctionTransform(columnName, mathFunction));
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
         * Convert an integer column to a set of 1 hot columns, based on the value in integer column
         *
         * @param columnName Name of the integer column
         * @param minValue   Minimum value possible for the integer column (inclusive)
         * @param maxValue   Maximum value possible for the integer column (inclusive)
         */
        public Builder integerToOneHot(String columnName, int minValue, int maxValue) {
            return transform(new IntegerToOneHotTransform(columnName, minValue, maxValue));
        }

        /**
         * Add a new column, where all values in the column are identical and as specified.
         *
         * @param newColumnName Name of the new column
         * @param newColumnType Type of the new column
         * @param fixedValue    Value in the new column for all records
         */
        public Builder addConstantColumn(String newColumnName, ColumnType newColumnType, Writable fixedValue) {
            return transform(new AddConstantColumnTransform(newColumnName, newColumnType, fixedValue));
        }

        /**
         * Add a new double column, where the value for that column (for all records) are identical
         *
         * @param newColumnName Name of the new column
         * @param value         Value in the new column for all records
         */
        public Builder addConstantDoubleColumn(String newColumnName, double value) {
            return addConstantColumn(newColumnName, ColumnType.Double, new DoubleWritable(value));
        }

        /**
         * Add a new integer column, where th
         * e value for that column (for all records) are identical
         *
         * @param newColumnName Name of the new column
         * @param value         Value of the new column for all records
         */
        public Builder addConstantIntegerColumn(String newColumnName, int value) {
            return addConstantColumn(newColumnName, ColumnType.Integer, new IntWritable(value));
        }

        /**
         * Add a new integer column, where the value for that column (for all records) are identical
         *
         * @param newColumnName Name of the new column
         * @param value         Value in the new column for all records
         */
        public Builder addConstantLongColumn(String newColumnName, long value) {
            return addConstantColumn(newColumnName, ColumnType.Long, new LongWritable(value));
        }


        /**
         * Convert the specified column to a string.
         * @param inputColumn the input column to convert
         * @return builder pattern
         */
        public Builder convertToString(String inputColumn) {
            return transform(new ConvertToString(inputColumn));
        }


        /**
         * Convert the specified column to a double.
         * @param inputColumn the input column to convert
         * @return builder pattern
         */
        public Builder convertToDouble(String inputColumn) {
            return transform(new ConvertToDouble(inputColumn));
        }


        /**
         * Convert the specified column to an integer.
         * @param inputColumn the input column to convert
         * @return builder pattern
         */
        public Builder convertToInteger(String inputColumn) {
            return transform(new ConvertToInteger(inputColumn));
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
                throw new IllegalStateException(
                        "Column \"" + column + "\" analysis is not numerical. " + "Column is not numerical?");

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
                    double meanExMin;
                    if (ca.getCountTotal() - countMin == 0) {
                        if (ca.getCountTotal() == 0) {
                            log.warn("Normalizing with Log2MeanExcludingMin but 0 records present in analysis");
                        } else {
                            log.warn("Normalizing with Log2MeanExcludingMin but all records are the same value");
                        }
                        meanExMin = mean;
                    } else {
                        meanExMin = (mean * ca.getCountTotal() - countMin * min) / (ca.getCountTotal() - countMin);
                    }
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
         * Convert a set of independent records/examples into a sequence; each example is simply treated as a sequence
         * of length 1, without any join/group operations. Note that more commonly, joining/grouping is required;
         * use {@link #convertToSequence(List, SequenceComparator)} for this functionality
         *
         */
        public Builder convertToSequence() {
            actionList.add(new DataAction(new ConvertToSequence(true, null, null)));
            return this;
        }

        /**
         * Convert a set of independent records/examples into a sequence, where each sequence is grouped according to
         * one or more key values (i.e., the values in one or more columns)
         * Within each sequence, values are ordered using the provided {@link SequenceComparator}
         *
         * @param keyColumns  Column to use as a key (values with the same key will be combined into sequences)
         * @param comparator A SequenceComparator to order the values within each sequence (for example, by time or String order)
         */
        public Builder convertToSequence(List<String> keyColumns, SequenceComparator comparator) {
            actionList.add(new DataAction(new ConvertToSequence(keyColumns, comparator)));
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
         * SequenceTrimTranform removes the first or last N values in a sequence. Note that the resulting sequence
         * may be of length 0, if the input sequence is less than or equal to N.
         *
         * @param numStepsToTrim Number of time steps to trim from the sequence
         * @param trimFromStart  If true: Trim values from the start of the sequence. If false: trim values from the end.
         */
        public Builder trimSequence(int numStepsToTrim, boolean trimFromStart) {
            actionList.add(new DataAction(new SequenceTrimTransform(numStepsToTrim, trimFromStart)));
            return this;
        }

        /**
         * Perform a sequence of operation on the specified columns. Note that this also truncates sequences by the
         * specified offset amount by default. Use {@code transform(new SequenceOffsetTransform(...)} to change this.
         * See {@link SequenceOffsetTransform} for details on exactly what this operation does and how.
         *
         * @param columnsToOffset Columns to offset
         * @param offsetAmount    Amount to offset the specified columns by (positive offset: 'columnsToOffset' are
         *                        moved to later time steps)
         * @param operationType   Whether the offset should be done in-place or by adding a new column
         */
        public Builder offsetSequence(List<String> columnsToOffset, int offsetAmount,
                                      SequenceOffsetTransform.OperationType operationType) {
            return transform(new SequenceOffsetTransform(columnsToOffset, offsetAmount, operationType,
                    SequenceOffsetTransform.EdgeHandling.TrimSequence, null));
        }


        /**
         * Reduce (i.e., aggregate/combine) a set of examples (typically by key).
         * <b>Note</b>: In the current implementation, reduction operations can be performed only on standard (i.e., non-sequence) data
         *
         * @param reducer Reducer to use
         */
        public Builder reduce(IAssociativeReducer reducer) {
            actionList.add(new DataAction(reducer));
            return this;
        }

        /**
         * Reduce (i.e., aggregate/combine) a set of sequence examples - for each sequence individually.
         * <b>Note</b>: This method results in non-sequence data. If you would instead prefer sequences of length 1
         * after the reduction, use {@code transform(new ReduceSequenceTransform(reducer))}.
         *
         * @param reducer        Reducer to use to reduce each window
         */
        public Builder reduceSequence(IAssociativeReducer reducer) {
            actionList.add(new DataAction(new ReduceSequenceTransform(reducer)));
            convertFromSequence();
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
        public Builder reduceSequenceByWindow(IAssociativeReducer reducer, WindowFunction windowFunction) {
            actionList.add(new DataAction(new ReduceSequenceByWindowTransform(reducer, windowFunction)));
            return this;
        }

        /**
         * SequenceMovingWindowReduceTransform: Adds a new column, where the value is derived by:<br>
         * (a) using a window of the last N values in a single column,<br>
         * (b) Apply a reduction op on the window to calculate a new value<br>
         * for example, this transformer can be used to implement a simple moving average of the last N values,
         * or determine the minimum or maximum values in the last N time steps.
         * <p>
         * For example, for a simple moving average, length 20: {@code new SequenceMovingWindowReduceTransform("myCol", 20, ReduceOp.Mean)}
         *
         * @param columnName Column name to perform windowing on
         * @param lookback   Look back period for windowing
         * @param op         Reduction operation to perform on each window
         */
        public Builder sequenceMovingWindowReduce(String columnName, int lookback, ReduceOp op) {
            actionList.add(new DataAction(new SequenceMovingWindowReduceTransform(columnName, lookback, op)));
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
        public Builder calculateSortedRank(String newColumnName, String sortOnColumn, WritableComparator comparator,
                                           boolean ascending) {
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
         * Append a String to a specified column
         *
         * @param column      Column to append the value to
         * @param toAppend    String to append to the end of each writable
         */
        public Builder appendStringColumnTransform(String column, String toAppend) {
            return transform(new AppendStringColumnTransform(column, toAppend));
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
         * Replace the values in a specified column with a specified "yes" value, if some condition holds.
         * Replace it with a "no" value, otherwise.
         *
         * @param column    Column to operate on
         * @param yesVal  Value to use as replacement, if condition is satisfied
         * @param noVal  Value to use as replacement, if condition is not satisfied
         * @param condition Condition that must be satisfied for replacement
         */
        public Builder conditionalReplaceValueTransformWithDefault(String column, Writable yesVal, Writable noVal, Condition condition) {
            return transform(new ConditionalReplaceValueTransformWithDefault(column, yesVal, noVal, condition));
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
         * Replace one or more String values in the specified column that match regular expressions.
         * <p>
         * Keys in the map are the regular expressions; the Values in the map are their String replacements.
         * For example:
         * <blockquote>
         * <table cellpadding="2">
         * <tr>
         *      <th>Original</th>
         *      <th>Regex</th>
         *      <th>Replacement</th>
         *      <th>Result</th>
         * </tr>
         * <tr>
         *      <td>Data_Vec</td>
         *      <td>_</td>
         *      <td></td>
         *      <td>DataVec</td>
         * </tr>
         * <tr>
         *      <td>B1C2T3</td>
         *      <td>\\d</td>
         *      <td>one</td>
         *      <td>BoneConeTone</td>
         * </tr>
         * <tr>
         *      <td>'&nbsp&nbsp4.25&nbsp'</td>
         *      <td>^\\s+|\\s+$</td>
         *      <td></td>
         *      <td>'4.25'</td>
         * </tr>
         * </table>
         * </blockquote>
         *
         * @param columnName Name of the column in which to do replacement
         * @param mapping    Map of old values or regular expression to new values
         */
        public Builder replaceStringTransform(String columnName, Map<String, String> mapping) {
            return transform(new ReplaceStringTransform(columnName, mapping));
        }

        /**
         * Element-wise NDArray math operation (add, subtract, etc) on an NDArray column
         *
         * @param columnName Name of the NDArray column to perform the operation on
         * @param op         Operation to perform
         * @param value      Value for the operation
         */
        public Builder ndArrayScalarOpTransform(String columnName, MathOp op, double value) {
            return transform(new NDArrayScalarOpTransform(columnName, op, value));
        }

        /**
         * Perform an element wise mathematical operation (such as add, subtract, multiply) on NDArray columns.
         * The existing columns are unchanged, a new NDArray column is added
         *
         * @param newColumnName Name of the new NDArray column
         * @param mathOp        Operation to perform
         * @param columnNames   Name of the columns used as input to the operation
         */
        public Builder ndArrayColumnsMathOpTransform(String newColumnName, MathOp mathOp, String... columnNames) {
            return transform(new NDArrayColumnsMathOpTransform(newColumnName, mathOp, columnNames));
        }

        /**
         * Apply an element wise mathematical function (sin, tanh, abs etc) to an NDArray column. This operation is
         * performed in place.
         *
         * @param columnName   Name of the column to perform the operation on
         * @param mathFunction Mathematical function to apply
         */
        public Builder ndArrayMathFunctionTransform(String columnName, MathFunction mathFunction) {
            return transform(new NDArrayMathFunctionTransform(columnName, mathFunction));
        }

        /**
         * Calculate a distance (cosine similarity, Euclidean, Manhattan) on two equal-sized NDArray columns. This
         * operation adds a new Double column (with the specified name) with the result.
         *
         * @param newColumnName Name of the new column (result) to add
         * @param distance      Distance to apply
         * @param firstCol      first column to use in the distance calculation
         * @param secondCol     second column to use in the distance calculation
         */
        public Builder ndArrayDistanceTransform(String newColumnName, Distance distance, String firstCol,
                                                String secondCol) {
            return transform(new NDArrayDistanceTransform(newColumnName, distance, firstCol, secondCol));
        }

        /**
         * Create the TransformProcess object
         */
        public TransformProcess build() {
            return new TransformProcess(this);
        }


    }


}
