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

package org.datavec.api.transform.transform.condition;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.ColumnOp;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.List;

/**
 * Replace the value in a specified column with a new value taken from another column, if a condition is satisfied/true.<br>
 * Note that the condition can be any generic condition, including on other column(s), different to the column
 * that will be modified if the condition is satisfied/true.<br>
 * <p>
 * <b>Note</b>: For sequences, this
 * transform use the convention that
 * each step in the sequence is passed
 * to the condition,
 * and replaced (or not) separately (i.e., Condition.condition(List<Writable>)
 * is used on each time step individually)
 *
 * @author Alex Black
 * @see ConditionalReplaceValueTransform to do a conditional replacement with a fixed value (instead of a value from another column)
 */
@JsonIgnoreProperties({"columnToReplaceIdx", "sourceColumnIdx"})
@EqualsAndHashCode(exclude = {"columnToReplaceIdx", "sourceColumnIdx"})
@Data
public class ConditionalCopyValueTransform implements Transform, ColumnOp {

    private final String columnToReplace;
    private final String sourceColumn;
    private final Condition condition;
    private int columnToReplaceIdx = -1;
    private int sourceColumnIdx = -1;

    /**
     * @param columnToReplace Name of the column in which to replace the old value
     * @param sourceColumn    Name of the column to get the new value from
     * @param condition       Condition
     */
    public ConditionalCopyValueTransform(@JsonProperty("columnToReplace") String columnToReplace,
                    @JsonProperty("sourceColumn") String sourceColumn, @JsonProperty("condition") Condition condition) {
        this.columnToReplace = columnToReplace;
        this.sourceColumn = sourceColumn;
        this.condition = condition;
    }

    @Override
    public Schema transform(Schema inputSchema) {
        //Conditional copy should not change any of the metadata, under normal usage
        return inputSchema;
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        if (!inputSchema.hasColumn(columnToReplace))
            throw new IllegalStateException("Column \"" + columnToReplace + "\" not found in input schema");
        if (!inputSchema.hasColumn(sourceColumn))
            throw new IllegalStateException("Column \"" + sourceColumn + "\" not found in input schema");
        columnToReplaceIdx = inputSchema.getIndexOfColumn(columnToReplace);
        sourceColumnIdx = inputSchema.getIndexOfColumn(sourceColumn);
        condition.setInputSchema(inputSchema);
    }

    @Override
    public Schema getInputSchema() {
        return condition.getInputSchema();
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        if (condition.condition(writables)) {
            //Condition holds -> set new value
            List<Writable> newList = new ArrayList<>(writables);
            newList.set(columnToReplaceIdx, writables.get(sourceColumnIdx));
            return newList;
        } else {
            //Condition does not hold -> no change
            return writables;
        }
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        List<List<Writable>> out = new ArrayList<>();
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
        return null;
    }

    /**
     * Transform a sequence
     *
     * @param sequence
     */
    @Override
    public Object mapSequence(Object sequence) {
        List<?> seq = (List<?>) sequence;
        List<Object> ret = new ArrayList<>();
        for (Object step : seq) {
            ret.add(map(step));
        }
        return ret;
    }

    @Override
    public String toString() {
        return "ConditionalCopyValueTransform(replaceColumn=\"" + columnToReplace + "\",sourceColumn=" + sourceColumn
                        + ",condition=" + condition + ")";
    }

    /**
     * The output column name
     * after the operation has been applied
     *
     * @return the output column name
     */
    @Override
    public String outputColumnName() {
        return columnToReplace;
    }

    /**
     * The output column names
     * This will often be the same as the input
     *
     * @return the output column names
     */
    @Override
    public String[] outputColumnNames() {
        return new String[] {columnToReplace};
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
        return sourceColumn;
    }
}
