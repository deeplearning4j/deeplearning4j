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

package org.datavec.api.transform.transform.sequence;

import lombok.Data;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.metadata.*;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.writable.*;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonInclude;

import java.util.ArrayList;
import java.util.List;

/**
 * SequenceDifferenceTransform: for an input sequence, calculate the difference on one column.<br>
 * For each time t, calculate someColumn(t) - someColumn(t-s), where s >= 1 is the 'lookback' period.<br>
 * <br>
 * Note: at t=0 (i.e., the first step in a sequence; or more generally, for all times t < s), there is no previous value
 * from which to calculate the difference. The {@link FirstStepMode} enumeration provides the following ways of handling
 * these time steps:<br>
 * 1. Default: output = someColumn(t) - someColumn(max(t-s, 0))
 * 2. SpecifiedValue: output = someColumn(t) - someColumn(t-s) if t-s >= 0, or a custom Writable object (for example, a DoubleWritable(0)
 * or NullWritable).
 * <p>
 * Note: this is an <i>in-place</i> operation: i.e., the values in each column are modified. If the original values are
 * to be retained in the data set, first make a copy (for example, using {@link org.datavec.api.transform.TransformProcess.Builder#duplicateColumn(String, String)})
 * and apply the difference operation in-place on the copy.
 *
 * @author Alex Black
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonIgnoreProperties({"inputSchema", "columnType"})
@Data
public class SequenceDifferenceTransform implements Transform {


    public enum FirstStepMode {
        Default, SpecifiedValue
    }

    private final String columnName;
    private final String newColumnName;
    private final int lookback;
    private final FirstStepMode firstStepMode;
    private final Writable specifiedValueWritable;

    private Schema inputSchema;
    private ColumnType columnType;

    /**
     * Create a SequenceDifferenceTransform with default lookback of 1, and using FirstStepMode.Default.
     * Output column name is the same as the input column name.
     *
     * @param columnName Name of the column to perform the operation on.
     */
    public SequenceDifferenceTransform(String columnName) {
        this(columnName, columnName, 1, FirstStepMode.Default, null);
    }

    /**
     * Create a SequenceDifferenceTransform with default lookback of 1, and using FirstStepMode.Default,
     * where the output column name is specified
     *
     * @param columnName    Name of the column to perform the operation on.
     * @param newColumnName New name for the column. May be same as the origina lcolumn name
     * @param lookback      Lookback period, in number of time steps. Must be > 0
     */
    public SequenceDifferenceTransform(String columnName, String newColumnName, int lookback) {
        this(columnName, newColumnName, lookback, FirstStepMode.Default, null);
    }

    /**
     * Create a SequenceDifferenceTransform with default lookback of 1, and using FirstStepMode.Default,
     * where the output column name is specified
     *
     * @param columnName             Name of the column to perform the operation on.
     * @param newColumnName          New name for the column. May be same as the origina lcolumn name
     * @param lookback               Lookback period, in number of time steps. Must be > 0
     * @param firstStepMode          see {@link FirstStepMode}
     * @param specifiedValueWritable Must be null if using FirstStepMode.Default, or non-null if using FirstStepMode.SpecifiedValue
     */
    public SequenceDifferenceTransform(String columnName, String newColumnName, int lookback,
                    FirstStepMode firstStepMode, Writable specifiedValueWritable) {
        if (firstStepMode != FirstStepMode.SpecifiedValue && specifiedValueWritable != null) {
            throw new IllegalArgumentException("Specified value writable provided (" + specifiedValueWritable + ") but "
                            + "firstStepMode != FirstStepMode.SpecifiedValue");
        }
        if (firstStepMode == FirstStepMode.SpecifiedValue && specifiedValueWritable == null) {
            throw new IllegalArgumentException(
                            "Specified value writable is null but firstStepMode != FirstStepMode.SpecifiedValue");
        }
        if (lookback <= 0) {
            throw new IllegalArgumentException("Lookback period must be > 0. Got: lookback period = " + lookback);
        }

        this.columnName = columnName;
        this.newColumnName = newColumnName;
        this.lookback = lookback;
        this.firstStepMode = firstStepMode;
        this.specifiedValueWritable = specifiedValueWritable;
    }

    /**
     * The output column name
     * after the operation has been applied
     *
     * @return the output column name
     */
    @Override
    public String outputColumnName() {
        return columnName;
    }

    /**
     * The output column names
     * This will often be the same as the input
     *
     * @return the output column names
     */
    @Override
    public String[] outputColumnNames() {
        return new String[] {columnName()};
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


    @Override
    public Schema transform(Schema inputSchema) {
        if (!inputSchema.hasColumn(columnName)) {
            throw new IllegalStateException("Invalid input schema: does not have column with name \"" + columnName
                            + "\"\n. All schema names: " + inputSchema.getColumnNames());
        }
        if (!(inputSchema instanceof SequenceSchema)) {
            throw new IllegalStateException(
                            "Invalid input schema: expected a SequenceSchema, got " + inputSchema.getClass());
        }

        List<ColumnMetaData> newMeta = new ArrayList<>(inputSchema.numColumns());
        for (ColumnMetaData m : inputSchema.getColumnMetaData()) {
            if (columnName.equals(m.getName())) {
                switch (m.getColumnType()) {
                    case Integer:
                        newMeta.add(new IntegerMetaData(newColumnName));
                        break;
                    case Long:
                        newMeta.add(new LongMetaData(newColumnName));
                        break;
                    case Double:
                        newMeta.add(new DoubleMetaData(newColumnName));
                        break;
                    case Float:
                        newMeta.add(new FloatMetaData(newColumnName));
                        break;
                    case Time:
                        newMeta.add(new LongMetaData(newColumnName)); //not Time - time column isn't used for duration...
                        break;
                    case Categorical:
                    case Bytes:
                    case String:
                    case Boolean:
                    default:
                        throw new IllegalStateException(
                                        "Cannot perform sequence difference on column of type " + m.getColumnType());
                }
            } else {
                newMeta.add(m);
            }
        }

        return inputSchema.newSchema(newMeta);
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        if (!inputSchema.hasColumn(columnName)) {
            throw new IllegalStateException("Invalid input schema: does not have column with name \"" + columnName
                            + "\"\n. All schema names: " + inputSchema.getColumnNames());
        }

        this.columnType = inputSchema.getMetaData(columnName).getColumnType();
        this.inputSchema = inputSchema;
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        throw new UnsupportedOperationException(
                        "Only sequence operations are supported for SequenceDifferenceTransform."
                                        + " Attempting to apply SequenceDifferenceTransform on non-sequence data?");
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        int columnIdx = inputSchema.getIndexOfColumn(columnName);

        int numSteps = sequence.size();
        List<List<Writable>> out = new ArrayList<>();
        for (int i = 0; i < numSteps; i++) {
            List<Writable> timeStep = sequence.get(i);
            List<Writable> newTimeStep = new ArrayList<>(timeStep.size());
            for (int j = 0; j < timeStep.size(); j++) {
                if (j == columnIdx) {
                    if (j < lookback && firstStepMode == FirstStepMode.SpecifiedValue) {
                        newTimeStep.add(specifiedValueWritable);
                    } else {
                        Writable current = timeStep.get(j);
                        Writable past = sequence.get(Math.max(0, i - lookback)).get(j);
                        switch (columnType) {
                            case Integer:
                                newTimeStep.add(new IntWritable(current.toInt() - past.toInt()));
                                break;
                            case Double:
                                newTimeStep.add(new DoubleWritable(current.toDouble() - past.toDouble()));
                                break;
                            case Float:
                                newTimeStep.add(new FloatWritable(current.toFloat() - past.toFloat()));
                                break;
                            case Long:
                            case Time:
                                newTimeStep.add(new LongWritable(current.toLong() - past.toLong()));
                                break;
                            default:
                                throw new IllegalStateException(
                                                "Cannot perform sequence difference on column of type " + columnType);
                        }
                    }
                } else {
                    newTimeStep.add(timeStep.get(j));
                }
            }
            out.add(newTimeStep);
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
        throw new UnsupportedOperationException(
                        "Only sequence operations are supported for SequenceDifferenceTransform."
                                        + " Attempting to apply SequenceDifferenceTransform on non-sequence data?");
    }

    /**
     * Transform a sequence
     *
     * @param sequence
     */
    @Override
    public Object mapSequence(Object sequence) {
        throw new UnsupportedOperationException(
                        "Only sequence operations are supported for SequenceDifferenceTransform."
                                        + " Attempting to apply SequenceDifferenceTransform on non-sequence data?");
    }
}
