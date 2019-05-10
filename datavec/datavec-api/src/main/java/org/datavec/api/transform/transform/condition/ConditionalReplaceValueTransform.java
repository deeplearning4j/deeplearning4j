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
 * Replace the value in a specified column with a new value, if a condition is satisfied/true.<br>
 * Note that the condition can be any generic condition, including on other column(s), different to the column
 * that will be modified if the condition is satisfied/true.<br>
 * <p>
 * <b>Note</b>: For sequences, this transform use the convention that each step in the sequence is passed to the condition,
 * and replaced (or not) separately (i.e., Condition.condition(List<Writable>) is used on each time step individually)
 *
 * @author Alex Black
 * @see ConditionalCopyValueTransform to do a conditional replacement with a value taken from another column
 */
@JsonIgnoreProperties({"columnToReplaceIdx"})
@EqualsAndHashCode(exclude = {"columnToReplaceIdx"})
@Data
public class ConditionalReplaceValueTransform implements Transform, ColumnOp {

    private final String columnToReplace;
    private final Writable newValue;
    private final Condition condition;
    private int columnToReplaceIdx = -1;

    /**
     * @param columnToReplace Name of the column in which to replace the old value with 'newValue', if the condition holds
     * @param newValue        New value to use
     * @param condition       Condition
     */
    public ConditionalReplaceValueTransform(@JsonProperty("columnToReplace") String columnToReplace,
                    @JsonProperty("newValue") Writable newValue, @JsonProperty("condition") Condition condition) {
        this.columnToReplace = columnToReplace;
        this.newValue = newValue;
        this.condition = condition;
    }

    @Override
    public Schema transform(Schema inputSchema) {
        //Conditional replace should not change any of the metadata, under normal usage
        return inputSchema;
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        columnToReplaceIdx = inputSchema.getColumnNames().indexOf(columnToReplace);
        if (columnToReplaceIdx < 0) {
            throw new IllegalStateException("Column \"" + columnToReplace + "\" not found in input schema");
        }
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
            newList.set(columnToReplaceIdx, newValue);
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
        if (condition.condition(input))
            return newValue;
        return input;

    }

    /**
     * Transform a sequence
     *
     * @param sequence
     */
    @Override
    public Object mapSequence(Object sequence) {
        List<?> seq = (List<?>) sequence;
        List<Object> out = new ArrayList<>();
        for (Object step : seq) {
            out.add(map(step));
        }
        return out;
    }

    @Override
    public String toString() {
        return "ConditionalReplaceValueTransform(replaceColumn=\"" + columnToReplace + "\",newValue=" + newValue
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
        return new String[] {columnToReplace};
    }

    /**
     * Returns a singular column name
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String columnName() {
        return columnToReplace;
    }
}
