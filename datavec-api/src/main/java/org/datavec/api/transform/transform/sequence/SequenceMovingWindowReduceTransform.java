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

package org.datavec.api.transform.transform.sequence;

import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.DoubleMetaData;
import org.datavec.api.transform.metadata.IntegerMetaData;
import org.datavec.api.transform.reduce.ReductionUtils;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonInclude;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

/**
 * SequenceMovingWindowReduceTransform Adds a new column, where the value is derived by:<br>
 * (a) using a window of the last N values in a single column,<br>
 * (b) Apply a reduction op on the window to calculate a new value<br>
 * for example, this transformer can be used to implement a simple moving average of the last N values,
 * or determine the minimum or maximum values in the last N time steps.
 *
 * For a simple moving average, length 20: {@code new SequenceMovingWindowReduceTransform("myCol", 20, ReduceOp.Mean)}
 *
 * @author Alex Black
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonIgnoreProperties({"inputSchema"})
public class SequenceMovingWindowReduceTransform implements Transform {

    public enum EdgeCaseHandling {
        Default, SpecifiedValue
    }

    private final String columnName;
    private final String newColumnName;
    private final int lookback;
    private final ReduceOp op;
    private final EdgeCaseHandling edgeCaseHandling;
    private final Writable edgeCaseValue;
    private Schema inputSchema;

    public SequenceMovingWindowReduceTransform(String columnName, int lookback, ReduceOp op) {
        this(columnName, defaultOutputColumnName(columnName, lookback, op), lookback, op, EdgeCaseHandling.Default, null);
    }

    public SequenceMovingWindowReduceTransform(String columnName, String newColumnName, int lookback, ReduceOp op,
                                               EdgeCaseHandling firstStepMode, Writable edgeCaseValue) {
        this.columnName = columnName;
        this.newColumnName = newColumnName;
        this.lookback = lookback;
        this.op = op;
        this.edgeCaseHandling = firstStepMode;
        this.edgeCaseValue = edgeCaseValue;
    }

    public static String defaultOutputColumnName(String originalName, int lookback, ReduceOp op){
        return op + "(" + lookback + "," + originalName + ")";
    }

    @Override
    public Schema transform(Schema inputSchema) {
        if (inputSchema != null && !(inputSchema instanceof SequenceSchema)) {
            throw new IllegalArgumentException("Invalid input: input schema must be a SequenceSchema");
        }

        int colIdx = inputSchema.getIndexOfColumn(columnName);

        //Approach here: The reducer gives us a schema for one time step -> simply convert this to a sequence schema...
        List<ColumnMetaData> oldMeta = inputSchema.getColumnMetaData();
        List<ColumnMetaData> meta = new ArrayList<>(oldMeta);

        ColumnMetaData m;
        switch (op){
            case Min:
            case Max:
            case Range:
            case TakeFirst:
            case TakeLast:
                //Same type as input
                m = oldMeta.get(colIdx);
                m = m.clone();
                m.setName(newColumnName);
                break;
            case Prod:
            case Sum:
            case Mean:
            case Stdev:
                //Double type
                m = new DoubleMetaData(newColumnName);
                break;
            case Count:
            case CountUnique:
                //Integer type
                m = new IntegerMetaData(newColumnName);
                break;
            default:
                throw new UnsupportedOperationException("Unknown op type: " + op);
        }
        meta.add(m);

        return new SequenceSchema(meta);
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        this.inputSchema = inputSchema;
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        throw new UnsupportedOperationException("SequenceMovingWindowReduceTransform can only be applied on sequences");
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        int colIdx = inputSchema.getIndexOfColumn(columnName);
        ColumnType columnType = inputSchema.getType(colIdx);
        List<List<Writable>> out = new ArrayList<>(sequence.size());
        LinkedList<Writable> window = new LinkedList<>();
        for(int i=0; i<sequence.size(); i++ ){
            Writable current = sequence.get(i).get(colIdx);
            window.addLast(current);
            if(window.size() > lookback){
                window.removeFirst();
            }
            Writable reduced;
            if(i <= lookback && edgeCaseHandling == EdgeCaseHandling.SpecifiedValue){
                reduced = edgeCaseValue;
            } else {
                reduced = ReductionUtils.reduceColumn(op, columnType, window, false, null );
            }
            ArrayList<Writable> outThisStep = new ArrayList<>(sequence.get(i).size()+1);
            outThisStep.addAll(sequence.get(i));
            outThisStep.add(reduced);
            out.add(outThisStep);
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
        throw new UnsupportedOperationException("SequenceMovingWindowReduceTransform can only be applied to sequences");
    }

    /**
     * Transform a sequence
     *
     * @param sequence
     */
    @Override
    public Object mapSequence(Object sequence) {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    public String toString() {
        return "SequenceMovingWindowReduceTransform(columnName=\"" + columnName + "\",newColumnName=\"" + newColumnName
                + "\",lookback=" + lookback + ",op=" + op + ",edgeCaseHandling=" + edgeCaseHandling
                + (edgeCaseHandling == EdgeCaseHandling.SpecifiedValue ? ",edgeCaseValue=" + edgeCaseValue : "") + ")";
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
        return columnNames();
    }

    /**
     * Returns column names
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String[] columnNames() {
        return getInputSchema().getColumnNames().toArray(new String[getInputSchema().numColumns()]);
    }

    /**
     * Returns a singular column name
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String columnName() {
        return columnNames()[0];
    }

}
